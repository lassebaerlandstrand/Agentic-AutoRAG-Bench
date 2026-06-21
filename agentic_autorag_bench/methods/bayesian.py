"""Optuna TPE baseline.

Ask-and-tell loop so the async ``evaluator`` integrates naturally — pruned
trials don't block subsequent suggestions. Sampler state is pickled per-trial
so a kill/resume reproduces the unattended sequence (Optuna's default
storage doesn't persist sampler state).
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_autorag.config.models import ProjectConfig
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench.methods._logging import log_trial_banner
from agentic_autorag_bench.methods._sampler import sample_optuna
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("agentic_autorag_bench.run")

_STUDY_NAME = "agentic_autorag_bench_bayesian_v2"
_DB_NAME = "optuna.db"
_SAMPLER_PICKLE = "optuna_sampler.pkl"
_WALL_CLOCK_NAME = "wall_clock.json"

# Cap on infeasible ask/tell-PRUNED rounds per budget slot before giving up.
# Optuna excludes PRUNED trials from TPE's KDE fit (no objective value to
# place them on the good/bad split), so the surrogate is unchanged. The
# retry terminates because ``ask()`` advances the sampler's internal RNG,
# yielding a different draw each call. For narrow infeasibilities (one
# embed × one chunk-size pairing) this converges in a few attempts; 1000 is
# paranoid. The canonical alternative is ``TPESampler(constraints_func=…)``
# for true constraint-aware TPE — overkill for our small infeasible region.
MAX_RESAMPLE_ATTEMPTS = 1000


@dataclass
class BayesianSearch:
    """Optuna TPE over the project's SearchSpace, with per-trial sampler persistence.

    Each ``budget.max_trials`` slot is guaranteed to evaluate a
    validation-passing config: an invalid suggestion is reported back to
    Optuna via ``tell(state=PRUNED)`` (so TPE's surrogate learns to avoid
    that region) and the slot is re-asked, up to ``MAX_RESAMPLE_ATTEMPTS``
    times. This keeps the budget comparable to the agentic baseline, whose
    Proposer is constraint-aware. Per-trial evaluation failures (LLM crash,
    etc.) still consume their slot. ``extras`` surfaces ``n_validation_rejects``
    and ``n_pruned`` (sampler-side prunes vs. validator-side rejects) so the
    paper can report how often TPE proposed infeasible points.
    """

    project: ProjectConfig
    storage_dir: Path
    resume: bool = False
    name: str = "bayesian"
    deterministic: bool = False

    async def search(
        self,
        evaluator: Evaluator,
        budget: Budget,
        *,
        seed: int | None = None,
    ) -> SearchResult:
        import optuna
        from optuna.trial import TrialState

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if budget.max_trials is None:
            raise ValueError("Bayesian search requires budget.max_trials")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.storage_dir / _DB_NAME
        sampler_path = self.storage_dir / _SAMPLER_PICKLE
        history_path = RunLayout(base=self.storage_dir).history
        wall_clock_path = self.storage_dir / _WALL_CLOCK_NAME

        # Wiping stale state on a fresh start is the bench-level ``--clean``
        # flag's job (``_clear_output_root_for`` in run.py). We deliberately
        # do NOT wipe here, even when ``resume=False``: a user who passes
        # ``--no-clean`` without ``--resume`` keeps whatever was on disk
        # (this matches the pre-existing semantic and protects in-progress
        # runs from accidental data loss).

        sampler: optuna.samplers.BaseSampler
        if self.resume and sampler_path.exists():
            try:
                sampler = pickle.loads(sampler_path.read_bytes())
            except Exception:
                logger.warning("Failed to load pickled sampler; starting fresh", exc_info=True)
                sampler = optuna.samplers.TPESampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=f"sqlite:///{db_path}",
            study_name=_STUDY_NAME,
            load_if_exists=True,
        )

        history: list[HistoryEntry] = []
        prior_wall_s = 0.0
        if self.resume:
            # A Ctrl+C between ``study.ask()`` and ``study.tell()`` leaves a
            # RUNNING trial in sqlite that TPE would never fit on. Mark them
            # FAIL so resumed asks generate fresh trial ids without leftover
            # noise in ``study.get_trials()``.
            for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                study.tell(t, state=TrialState.FAIL)

            if history_path.exists():
                history = _load_history(history_path)
            else:
                # Self-heal: a run started with a pre-resume version of the
                # bench never wrote ``history.jsonl`` per trial, so resuming
                # it would lose the prior trials. Rebuild from optuna.db
                # (params + score) and trial_cost_ledger.jsonl (cost/tokens).
                # ``metrics`` other than ``answer_accuracy`` (which equals
                # score in bayesian) are not recoverable — they default to
                # zero, which only affects downstream plots for the
                # recovered trials, not the optimizer's continuation.
                history = _reconstruct_history_from_optuna(
                    study,
                    self.project,
                    self.storage_dir,
                )
                for entry in history:
                    _append_history(history_path, entry)
                if history:
                    logger.warning(
                        "Reconstructed %d trial(s) into a fresh history.jsonl "
                        "from optuna.db + trial_cost_ledger.jsonl. "
                        "Per-trial ``metrics`` other than score default to 0 "
                        "for these trials (the originals were never persisted "
                        "to disk); downstream plots will show zeros for "
                        "mean_em / mean_f1 / mean_retrieval_quality.",
                        len(history),
                    )

            if wall_clock_path.exists():
                try:
                    raw = json.loads(wall_clock_path.read_text(encoding="utf-8"))
                    prior_wall_s = float(raw.get("wall_clock_s", 0.0))
                except Exception:
                    logger.warning(
                        "Could not parse %s; wall-clock starts at 0",
                        wall_clock_path,
                        exc_info=True,
                    )
            logger.info(
                "Resuming Bayesian search from trial %d/%d (prior wall=%.1fs)",
                len(history) + 1,
                budget.max_trials,
                prior_wall_s,
            )

        trial_usd_total = sum(h.eval_usd for h in history)
        n_validation_rejects = 0
        n_pruned = 0

        t_start = time.monotonic()
        for trial_num in range(len(history) + 1, budget.max_trials + 1):
            trial = None
            config = None
            for _ in range(MAX_RESAMPLE_ATTEMPTS):
                candidate_trial = study.ask()
                try:
                    candidate = sample_optuna(candidate_trial, self.project.search_space)
                except optuna.TrialPruned as exc:
                    logger.debug("trial %d resample pruned: %s", trial_num, exc)
                    study.tell(candidate_trial, state=optuna.trial.TrialState.PRUNED)
                    n_pruned += 1
                    continue

                violations = self.project.validate_trial(candidate)
                if not violations:
                    trial = candidate_trial
                    config = candidate
                    break
                logger.debug("trial %d resample rejected: %s", trial_num, "; ".join(violations))
                study.tell(candidate_trial, state=optuna.trial.TrialState.PRUNED)
                n_validation_rejects += 1

            if trial is None or config is None:
                raise RuntimeError(
                    f"Bayesian search could not find a valid config after "
                    f"{MAX_RESAMPLE_ATTEMPTS} Optuna ask/tell rounds on trial "
                    f"{trial_num}; TPE failed to learn the feasible region, "
                    f"which usually means the search space's feasible volume "
                    f"is near zero."
                )

            log_trial_banner(logger, trial_num, budget.max_trials, config)

            try:
                result = await evaluator(config)
            except Exception:
                logger.exception("trial %d evaluation failed; marking failed", trial_num)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue

            study.tell(trial, result.answer_accuracy)
            entry = HistoryEntry(
                trial_number=trial_num,
                config=config.to_prompt_dump(include_graph=self.project.uses_graph()),
                answer_accuracy=result.answer_accuracy,
                metrics=result.metrics,
                eval_usd=result.eval_usd,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                embedding_tokens=result.embedding_tokens,
            )
            history.append(entry)
            trial_usd_total += result.eval_usd

            try:
                sampler_path.write_bytes(pickle.dumps(study.sampler))
            except Exception:
                logger.warning("Failed to persist Optuna sampler", exc_info=True)
            _append_history(history_path, entry)
            cumulative = prior_wall_s + (time.monotonic() - t_start)
            wall_clock_path.write_text(json.dumps({"wall_clock_s": cumulative}), encoding="utf-8")

            best = max(h.answer_accuracy for h in history)
            logger.info(
                "bayesian trial %d done | accuracy=%.3f | best so far=%.3f", trial_num, result.answer_accuracy, best
            )
            logger.info("")

        if not history:
            raise RuntimeError("Bayesian search produced no successful trials")

        best_entry = max(history, key=lambda h: h.answer_accuracy)
        total_wall = prior_wall_s + (time.monotonic() - t_start)
        return SearchResult(
            method=self.name,
            seed=seed,
            deterministic=self.deterministic,
            best_config=best_entry.config,
            history=history,
            optimizer_usd=0.0,
            trial_usd_total=trial_usd_total,
            wall_clock_s=total_wall,
            prompt_tokens=sum(h.prompt_tokens for h in history),
            completion_tokens=sum(h.completion_tokens for h in history),
            embedding_tokens=sum(h.embedding_tokens for h in history),
            extras={
                "n_validation_rejects": n_validation_rejects,
                "n_pruned": n_pruned,
                "study_name": _STUDY_NAME,
                "storage": str(db_path),
            },
        )


def _load_history(path: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        entries.append(HistoryEntry(**data))
    return entries


def _append_history(path: Path, entry: HistoryEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict()) + "\n")


def _reconstruct_history_from_optuna(study, project, storage_dir: Path) -> list[HistoryEntry]:
    """Rebuild a ``HistoryEntry`` list from ``optuna.db`` + ``trial_cost_ledger.jsonl``.

    Used as a self-heal path on ``--resume`` when ``history.jsonl`` is
    missing — typically because the run was started with a pre-resume
    bench that only wrote history at end-of-method. Each completed Optuna
    trial's stored params + value give us the ``TrialConfig`` (via
    ``sample_optuna`` replay against the FrozenTrial) and ``score``;
    ``trial_cost_ledger.jsonl`` gives us per-trial cost + token totals.
    ``metrics`` other than ``answer_accuracy`` (which equals score in
    Bayesian) are not recoverable and default to zero.

    Bench's per-iteration ``trial_num`` (1-indexed, increments only on
    SUCCESSFUL evaluation) corresponds to the sorted order of COMPLETE
    Optuna trials, NOT to ``FrozenTrial.number`` (which also increments
    on pruned/failed asks). That's why we enumerate completed trials in
    optuna-number order and assign bench indices 1..N.
    """
    import optuna  # local to keep the import out of this module's top-level hot path

    from agentic_autorag_bench.methods._sampler import sample_optuna

    completed = sorted(
        study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)),
        key=lambda t: t.number,
    )
    if not completed:
        return []

    cost_path = RunLayout(base=storage_dir).trial_cost_ledger
    cost_by_trial: dict[int, dict[str, float | int]] = {}
    if cost_path.exists():
        for line in cost_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            trial_num = int(rec.get("trial_number", 0))
            buckets = rec.get("buckets", {})
            agg = {
                "eval_usd": sum(float(b.get("usd", 0.0)) for b in buckets.values()),
                "prompt_tokens": sum(int(b.get("prompt_tokens", 0)) for b in buckets.values()),
                "completion_tokens": sum(int(b.get("completion_tokens", 0)) for b in buckets.values()),
                "embedding_tokens": sum(int(b.get("embedding_input_tokens", 0)) for b in buckets.values()),
            }
            cost_by_trial[trial_num] = agg

    include_graph = project.uses_graph()
    entries: list[HistoryEntry] = []
    for bench_idx, ft in enumerate(completed, start=1):
        try:
            cfg = sample_optuna(ft, project.search_space)
        except Exception:
            logger.warning(
                "Could not reconstruct config for optuna trial %d during resume; skipping",
                ft.number,
                exc_info=True,
            )
            continue
        score = float(ft.value) if ft.value is not None else 0.0
        agg = cost_by_trial.get(bench_idx, {})
        entries.append(
            HistoryEntry(
                trial_number=bench_idx,
                config=cfg.to_prompt_dump(include_graph=include_graph),
                answer_accuracy=score,
                metrics={
                    "answer_accuracy": score,
                    "mean_em": 0.0,
                    "mean_f1": 0.0,
                    "mean_retrieval_quality": 0.0,
                },
                eval_usd=float(agg.get("eval_usd", 0.0)),
                prompt_tokens=int(agg.get("prompt_tokens", 0)),
                completion_tokens=int(agg.get("completion_tokens", 0)),
                embedding_tokens=int(agg.get("embedding_tokens", 0)),
            )
        )
    return entries

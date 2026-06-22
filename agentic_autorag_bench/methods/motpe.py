"""Optuna MO-TPE baseline (syftr-style).

The optimizer syftr (Conway et al., AutoML-Conf 2025) runs: group-decomposed
multivariate TPE with ``constant_liar``, single-objective in ``maximize`` mode
for the accuracy experiment and two-objective (accuracy ↑, per-query LLM cost ↓)
for the Pareto experiment. We replicate the *sampler* inside the bench harness so
the comparison is fair by construction — identical search space, exam, evaluator,
budget and seeds; only the proposer differs (YAHPO/HPOBench standard).

``group=True`` is mandatory, not optional: our search space is define-by-run
conditional (``hybrid_alpha`` only under hybrid+alpha, ``reranker_top_n`` gated on
a reranker, graph fields skipped off-graph, per-trial dynamic int bounds). Optuna's
plain ``multivariate=True, group=False`` does not support dynamic search spaces;
``group=True`` (which forces multivariate) decomposes the space into the sub-spaces
each trial actually visits. This is syftr's ``HierarchicalTPESampler`` intent.

Single- vs multi-objective is config-driven (``meta.cost_aware``), mirroring how
the agentic optimizer flips ``agentic_score`` → ``agentic_cost`` on the same flag.
The ``warm_start`` variant seeds the first ``n_startup_trials`` slots with cold
KB-grounded proposer configs instead of random draws (see ``_propose_warmstart``).

Ask-and-tell loop so the async ``evaluator`` integrates naturally — pruned trials
don't block subsequent suggestions. Sampler state is pickled per-trial so a
kill/resume reproduces the unattended sequence (Optuna's default storage doesn't
persist sampler state).
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
from agentic_autorag_bench.methods._sampler import config_to_optuna_params, sample_optuna
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("agentic_autorag_bench.run")

_STUDY_NAME = "agentic_autorag_bench_motpe_v1"
_DB_NAME = "optuna.db"
_SAMPLER_PICKLE = "optuna_sampler.pkl"
_WALL_CLOCK_NAME = "wall_clock.json"

# syftr's published sampler configuration (syftr/optuna_helper.py). Exposed as a
# dict so the behavioral-equivalence test can assert our construction matches it
# without poking at TPESampler internals. ``n_startup_trials`` is Optuna's library
# default (10); syftr's 100 is sized for its 1000-trial budget — not comparable at
# 40 trials. Disclosed in the paper; identical across Exp 1 / Exp 2 and all seeds.
N_STARTUP_TRIALS = 10
SAMPLER_KWARGS: dict[str, object] = {
    "multivariate": True,
    "group": True,
    "constant_liar": True,
    "n_startup_trials": N_STARTUP_TRIALS,
}

# Cap on infeasible ask/tell-PRUNED rounds per budget slot before giving up.
# Optuna excludes PRUNED trials from TPE's KDE fit (no objective value to place
# them on the good/bad split), so the surrogate is unchanged. The retry
# terminates because ``ask()`` advances the sampler's internal RNG, yielding a
# different draw each call. For narrow infeasibilities this converges in a few
# attempts; 1000 is paranoid.
MAX_RESAMPLE_ATTEMPTS = 1000


def _make_sampler(seed: int | None):
    import optuna

    return optuna.samplers.TPESampler(seed=seed, **SAMPLER_KWARGS)


@dataclass
class MOTPESearch:
    """Group-decomposed multivariate MO-TPE over the project's SearchSpace.

    Single-objective (``direction="maximize"``) when ``project.meta.cost_aware``
    is false; two-objective (``directions=["maximize", "minimize"]`` over
    ``answer_accuracy`` and ``mean_llm_cost_per_query_usd``) when it is true.

    Each ``budget.max_trials`` slot is guaranteed to evaluate a
    validation-passing config: an invalid suggestion is reported back to Optuna
    via ``tell(state=PRUNED)`` (so TPE's surrogate learns to avoid that region)
    and the slot is re-asked, up to ``MAX_RESAMPLE_ATTEMPTS`` times. This keeps
    the budget comparable to the agentic baseline, whose Proposer is
    constraint-aware. ``extras`` surfaces ``n_validation_rejects`` and
    ``n_pruned`` so the paper can report how often TPE proposed infeasible points.

    ``warm_start`` replaces the random startup trials with cold KB-grounded
    proposer configs (the agent's frozen prior); the proposer spend is reported
    under ``optimizer_usd``.
    """

    project: ProjectConfig
    storage_dir: Path
    resume: bool = False
    warm_start: bool = False
    name: str = "motpe"
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
            raise ValueError("MO-TPE search requires budget.max_trials")

        cost_aware = bool(self.project.meta.cost_aware)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.storage_dir / _DB_NAME
        sampler_path = self.storage_dir / _SAMPLER_PICKLE
        history_path = RunLayout(base=self.storage_dir).history
        wall_clock_path = self.storage_dir / _WALL_CLOCK_NAME

        # Wiping stale state on a fresh start is the bench-level ``--clean``
        # flag's job (``_clear_output_root_for`` in run.py). We deliberately do
        # NOT wipe here, even when ``resume=False``: a user who passes
        # ``--no-clean`` without ``--resume`` keeps whatever was on disk.

        sampler: optuna.samplers.BaseSampler
        if self.resume and sampler_path.exists():
            try:
                sampler = pickle.loads(sampler_path.read_bytes())
            except Exception:
                logger.warning("Failed to load pickled sampler; starting fresh", exc_info=True)
                sampler = _make_sampler(seed)
        else:
            sampler = _make_sampler(seed)

        if cost_aware:
            study = optuna.create_study(
                directions=["maximize", "minimize"],
                sampler=sampler,
                storage=f"sqlite:///{db_path}",
                study_name=_STUDY_NAME,
                load_if_exists=True,
            )
        else:
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
                history = _reconstruct_history_from_optuna(study, self.project, self.storage_dir)
                for entry in history:
                    _append_history(history_path, entry)
                if history:
                    logger.warning(
                        "Reconstructed %d trial(s) into a fresh history.jsonl from "
                        "optuna.db + trial_cost_ledger.jsonl. Per-trial ``metrics`` "
                        "other than accuracy default to 0 for these trials.",
                        len(history),
                    )

            if wall_clock_path.exists():
                try:
                    raw = json.loads(wall_clock_path.read_text(encoding="utf-8"))
                    prior_wall_s = float(raw.get("wall_clock_s", 0.0))
                except Exception:
                    logger.warning("Could not parse %s; wall-clock starts at 0", wall_clock_path, exc_info=True)
            logger.info(
                "Resuming MO-TPE search from trial %d/%d (prior wall=%.1fs)",
                len(history) + 1,
                budget.max_trials,
                prior_wall_s,
            )

        trial_usd_total = sum(h.eval_usd for h in history)
        optimizer_usd = 0.0
        n_validation_rejects = 0
        n_pruned = 0
        n_warmstart = 0

        # Warm-start: enqueue cold KB-grounded proposer configs so the first
        # ``n_startup`` asks return the agent's frozen prior instead of random
        # draws. Only on a fresh start — on resume the enqueued WAITING trials
        # persist in storage and would otherwise be double-queued.
        if self.warm_start and not self.resume and not history:
            warm_configs, optimizer_usd = await self._propose_warmstart(N_STARTUP_TRIALS, seed)
            for cfg in warm_configs:
                study.enqueue_trial(config_to_optuna_params(cfg, self.project.search_space), skip_if_exists=False)
            n_warmstart = len(warm_configs)
            logger.info("MO-TPE warm-start: enqueued %d cold proposer config(s)", n_warmstart)

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
                    f"MO-TPE search could not find a valid config after "
                    f"{MAX_RESAMPLE_ATTEMPTS} Optuna ask/tell rounds on trial "
                    f"{trial_num}; the search space's feasible volume is near zero."
                )

            log_trial_banner(logger, trial_num, budget.max_trials, config)

            try:
                result = await evaluator(config)
            except Exception:
                logger.exception("trial %d evaluation failed; marking failed", trial_num)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue

            if cost_aware:
                study.tell(trial, [result.answer_accuracy, result.mean_llm_cost_per_query_usd])
            else:
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
                mean_llm_cost_per_query_usd=result.mean_llm_cost_per_query_usd,
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
                "%s trial %d done | accuracy=%.3f | best so far=%.3f",
                self.name,
                trial_num,
                result.answer_accuracy,
                best,
            )
            logger.info("")

        if not history:
            raise RuntimeError("MO-TPE search produced no successful trials")

        # best_config is the max-accuracy point in both modes — matching what the
        # agentic_cost optimizer picks for its held-out eval. The Pareto figure
        # reads the whole frontier from history separately (per-query cost is on
        # every HistoryEntry).
        best_entry = max(history, key=lambda h: h.answer_accuracy)
        total_wall = prior_wall_s + (time.monotonic() - t_start)
        return SearchResult(
            method=self.name,
            seed=seed,
            deterministic=self.deterministic,
            best_config=best_entry.config,
            history=history,
            optimizer_usd=optimizer_usd,
            trial_usd_total=trial_usd_total,
            wall_clock_s=total_wall,
            prompt_tokens=sum(h.prompt_tokens for h in history),
            completion_tokens=sum(h.completion_tokens for h in history),
            embedding_tokens=sum(h.embedding_tokens for h in history),
            extras={
                "n_validation_rejects": n_validation_rejects,
                "n_pruned": n_pruned,
                "n_warmstart": n_warmstart,
                "cost_aware": cost_aware,
                "study_name": _STUDY_NAME,
                "storage": str(db_path),
            },
        )

    async def _propose_warmstart(self, n: int, seed: int | None) -> tuple[list, float]:
        """Generate up to ``n`` distinct cold proposer configs (the agent's frozen
        prior) and return ``(configs, proposer_usd)``.

        The same KB-grounded optimizer LLM the agent uses, called with empty trial
        history so each config reflects only the KB + corpus description + search
        space — no per-question reasoning. ``meta.cost_aware`` drives the prompt
        stance so the warm-start prior matches the experiment (cost-aware for the
        Pareto run). The proposer spend is captured from the active cost-ledger
        delta and reported as ``optimizer_usd`` (optimizer-side, like the agent's
        own proposer), not folded into any trial.
        """
        from agentic_autorag.config.knowledge_base import KnowledgeBase
        from agentic_autorag.cost_ledger import get_active_ledger
        from agentic_autorag.optimizer.history import HistoryLog
        from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent

        try:
            kb: KnowledgeBase | None = KnowledgeBase()
        except Exception:
            logger.warning("Warm-start: could not load knowledge base; proposing without it", exc_info=True)
            kb = None

        history = HistoryLog(path=str(self.storage_dir / "_warmstart_history.jsonl"), load_existing=False)
        agent = ReasoningAgent(
            agent_model=self.project.agent.optimizer_model,
            config=self.project,
            history=history,
            knowledge_base=kb,
            seed=seed,
        )
        corpus_description = self.project.meta.corpus_description
        include_graph = self.project.uses_graph()

        ledger = get_active_ledger()
        before = ledger.snapshot() if ledger is not None else None

        configs: list = []
        seen: set[str] = set()
        base = (seed if seed is not None else 0) * 1000
        attempts = 0
        max_attempts = n * 5
        while len(configs) < n and attempts < max_attempts:
            # Vary the seed per call so a seed-respecting provider still yields a
            # diverse draw from the prior (a seed-ignoring provider varies via
            # intrinsic nondeterminism either way).
            agent.seed = base + attempts
            attempts += 1
            try:
                cfg = await agent.propose_initial(corpus_description)
            except Exception:
                logger.warning("Warm-start proposal attempt failed; continuing", exc_info=True)
                continue
            key = json.dumps(cfg.to_prompt_dump(include_graph=include_graph), sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            configs.append(cfg)

        if len(configs) < n:
            logger.warning(
                "Warm-start produced %d/%d distinct configs after %d attempts; "
                "the remaining startup slots fall back to TPE random draws.",
                len(configs),
                n,
                attempts,
            )

        proposer_usd = 0.0
        if ledger is not None and before is not None:
            delta = ledger.delta_since(before)
            proposer_usd = sum(float(b.get("usd", 0.0)) for b in delta.values())
        return configs, proposer_usd


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

    Used as a self-heal path on ``--resume`` when ``history.jsonl`` is missing.
    Each completed Optuna trial's stored params + objective give the
    ``TrialConfig`` (via ``sample_optuna`` replay) and ``answer_accuracy``
    (objective 0 in both single- and multi-objective mode);
    ``trial_cost_ledger.jsonl`` gives per-trial cost + token totals. ``metrics``
    other than accuracy are not recoverable and default to zero.

    Bench's per-iteration ``trial_num`` (1-indexed, increments only on SUCCESSFUL
    evaluation) corresponds to the sorted order of COMPLETE Optuna trials, NOT to
    ``FrozenTrial.number`` (which also increments on pruned/failed asks).
    """
    import optuna

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

    # Multi-objective (cost_aware) studies store the per-query cost as objective-1
    # (``study.tell(trial, [accuracy, mean_llm_cost_per_query_usd])``). Recover it
    # so resumed Pareto trials keep the SAME cost objective the live path records —
    # it is unrecoverable from trial_cost_ledger.jsonl (which holds judge-inclusive
    # bucket totals, not the per-query deploy-time mean). Single-objective studies
    # have no cost to recover; 0.0 is correct there.
    n_objectives = len(study.directions)
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
        # ``values[0]`` is accuracy in both single- and multi-objective mode
        # (``FrozenTrial.value`` raises for multi-objective trials).
        score = float(ft.values[0]) if ft.values else 0.0
        cost = float(ft.values[1]) if (n_objectives > 1 and ft.values and len(ft.values) > 1) else 0.0
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
                mean_llm_cost_per_query_usd=cost,
            )
        )
    return entries

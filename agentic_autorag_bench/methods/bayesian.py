"""Optuna TPE baseline.

Ask-and-tell loop so the async ``evaluator`` integrates naturally — pruned
trials don't block subsequent suggestions. Sampler state is pickled per-trial
so a kill/resume reproduces the unattended sequence (Optuna's default
storage doesn't persist sampler state).
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_autorag.config.models import ProjectConfig

from agentic_autorag_bench.methods._logging import log_trial_banner
from agentic_autorag_bench.methods._sampler import sample_optuna
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("agentic_autorag_bench.run")

_STUDY_NAME = "agentic_autorag_bench_bayesian_v2"
_DB_NAME = "optuna.db"
_SAMPLER_PICKLE = "optuna_sampler.pkl"

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

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if budget.max_trials is None:
            raise ValueError("Bayesian search requires budget.max_trials")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.storage_dir / _DB_NAME
        sampler_path = self.storage_dir / _SAMPLER_PICKLE

        sampler: optuna.samplers.BaseSampler
        if sampler_path.exists():
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
        trial_usd_total = 0.0
        n_validation_rejects = 0
        n_pruned = 0

        t_start = time.monotonic()
        for trial_num in range(1, budget.max_trials + 1):
            trial = None
            config = None
            for _ in range(MAX_RESAMPLE_ATTEMPTS):
                candidate_trial = study.ask()
                try:
                    candidate = sample_optuna(
                        candidate_trial,
                        self.project.search_space,
                        self.project.embedding_token_limits,
                    )
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

            study.tell(trial, result.score)
            history.append(
                HistoryEntry(
                    trial_number=trial_num,
                    config=config.to_prompt_dump(include_graph=self.project.uses_graph()),
                    score=result.score,
                    metrics=result.metrics,
                    eval_usd=result.eval_usd,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    embedding_tokens=result.embedding_tokens,
                )
            )
            trial_usd_total += result.eval_usd

            try:
                sampler_path.write_bytes(pickle.dumps(study.sampler))
            except Exception:
                logger.warning("Failed to persist Optuna sampler", exc_info=True)

            best = max(h.score for h in history)
            logger.info("bayesian trial %d done | score=%.3f | best so far=%.3f", trial_num, result.score, best)
            logger.info("")

        if not history:
            raise RuntimeError("Bayesian search produced no successful trials")

        best_entry = max(history, key=lambda h: h.score)
        return SearchResult(
            method=self.name,
            seed=seed,
            deterministic=self.deterministic,
            best_config=best_entry.config,
            history=history,
            optimizer_usd=0.0,
            trial_usd_total=trial_usd_total,
            wall_clock_s=time.monotonic() - t_start,
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

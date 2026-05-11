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

from autorag_bench.methods._sampler import sample_optuna
from autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("autorag_bench.run")

_STUDY_NAME = "autorag_bench_bayesian"
_DB_NAME = "optuna.db"
_SAMPLER_PICKLE = "optuna_sampler.pkl"


@dataclass
class BayesianSearch:
    """Optuna TPE over the project's SearchSpace, with per-trial sampler persistence."""

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
            trial = study.ask()
            try:
                config = sample_optuna(trial, self.project.search_space, self.project.embedding_token_limits)
            except optuna.TrialPruned as exc:
                logger.warning("trial %d pruned during sampling: %s", trial_num, exc)
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                n_pruned += 1
                continue

            violations = self.project.validate_trial(config)
            if violations:
                logger.warning("trial %d rejected: %s", trial_num, "; ".join(violations))
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                n_validation_rejects += 1
                continue

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
                )
            )
            trial_usd_total += result.eval_usd

            try:
                sampler_path.write_bytes(pickle.dumps(study.sampler))
            except Exception:
                logger.warning("Failed to persist Optuna sampler", exc_info=True)

            best = max(h.score for h in history)
            logger.info("bayesian trial %d done | score=%.3f | best so far=%.3f", trial_num, result.score, best)

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
            extras={
                "n_validation_rejects": n_validation_rejects,
                "n_pruned": n_pruned,
                "study_name": _STUDY_NAME,
                "storage": str(db_path),
            },
        )

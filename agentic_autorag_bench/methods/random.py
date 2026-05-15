"""Uniform-random search baseline."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from agentic_autorag.config.models import ProjectConfig

from agentic_autorag_bench.methods._logging import log_trial_banner
from agentic_autorag_bench.methods._sampler import sample_random
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("agentic_autorag_bench.run")


@dataclass
class RandomSearch:
    """Uniformly samples ``TrialConfig`` from the project's ``SearchSpace``.

    Each iteration of the loop occupies one slot of ``budget.max_trials``.
    Validation rejects and per-trial evaluation failures still consume their
    slot (``continue`` skips the score-record but does not re-sample): the
    budget reflects work attempted, not work that succeeded. ``extras``
    surfaces ``n_validation_rejects`` so the paper can report the count.
    """

    project: ProjectConfig
    name: str = "random"
    deterministic: bool = False

    async def search(
        self,
        evaluator: Evaluator,
        budget: Budget,
        *,
        seed: int | None = None,
    ) -> SearchResult:
        if budget.max_trials is None:
            raise ValueError("Random search requires budget.max_trials")

        rng = random.Random(seed if seed is not None else 0)
        history: list[HistoryEntry] = []
        trial_usd_total = 0.0
        n_validation_rejects = 0

        t_start = time.monotonic()
        for trial_num in range(1, budget.max_trials + 1):
            config = sample_random(rng, self.project.search_space, self.project.embedding_token_limits)
            violations = self.project.validate_trial(config)
            if violations:
                logger.warning("trial %d rejected: %s", trial_num, "; ".join(violations))
                n_validation_rejects += 1
                continue

            log_trial_banner(logger, trial_num, budget.max_trials, config)

            try:
                result = await evaluator(config)
            except Exception:
                logger.exception("trial %d evaluation failed; skipping", trial_num)
                continue

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
            best = max(h.score for h in history)
            logger.info("random trial %d done | score=%.3f | best so far=%.3f", trial_num, result.score, best)
            logger.info("")

        if not history:
            raise RuntimeError("Random search produced no successful trials")

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
            extras={"n_validation_rejects": n_validation_rejects},
        )

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

# Cap on how many invalid samples one trial may discard before giving up.
# A search space where 1000 uniform draws can't find a feasible config is
# either misconfigured or has near-zero feasible volume; either way, raising
# is more honest than silently emitting an invalid trial.
MAX_RESAMPLE_ATTEMPTS = 1000


@dataclass
class RandomSearch:
    """Uniformly samples ``TrialConfig`` from the project's ``SearchSpace``.

    Each iteration of the loop occupies one slot of ``budget.max_trials`` and
    is guaranteed to evaluate a validation-passing config: invalid draws (e.g.
    ``chunk_token_size`` exceeding the sampled embedding's context window when
    the size grid has no feasible value) are discarded and resampled, up to
    ``MAX_RESAMPLE_ATTEMPTS`` per trial. This makes the budget comparable to
    the agentic baseline, whose Proposer is constraint-aware and never spends
    a trial on an infeasible config. Per-trial evaluation failures (LLM
    crash, etc.) still consume their slot — those are not a sampler artefact.
    ``extras`` surfaces ``n_validation_rejects`` so the paper can report how
    "narrow" the feasible region was.
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
            config = None
            for _ in range(MAX_RESAMPLE_ATTEMPTS):
                candidate = sample_random(rng, self.project.search_space)
                violations = self.project.validate_trial(candidate)
                if not violations:
                    config = candidate
                    break
                n_validation_rejects += 1
                logger.debug("trial %d resample: %s", trial_num, "; ".join(violations))
            if config is None:
                raise RuntimeError(
                    f"Random search could not find a valid config after "
                    f"{MAX_RESAMPLE_ATTEMPTS} resamples on trial {trial_num}; "
                    f"the feasible region of the search space is too narrow."
                )

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
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    embedding_tokens=result.embedding_tokens,
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
            prompt_tokens=sum(h.prompt_tokens for h in history),
            completion_tokens=sum(h.completion_tokens for h in history),
            embedding_tokens=sum(h.embedding_tokens for h in history),
            extras={"n_validation_rejects": n_validation_rejects},
        )

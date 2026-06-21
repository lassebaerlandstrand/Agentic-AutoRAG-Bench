"""Uniform-random search baseline."""

from __future__ import annotations

import json
import logging
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_autorag.config.models import ProjectConfig
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench.methods._logging import log_trial_banner
from agentic_autorag_bench.methods._sampler import sample_random
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("agentic_autorag_bench.run")

# Cap on how many invalid samples one trial may discard before giving up.
# A search space where 1000 uniform draws can't find a feasible config is
# either misconfigured or has near-zero feasible volume; either way, raising
# is more honest than silently emitting an invalid trial.
MAX_RESAMPLE_ATTEMPTS = 1000

_RNG_STATE_NAME = "rng_state.pkl"
_WALL_CLOCK_NAME = "wall_clock.json"


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

    With a ``storage_dir`` set, each completed trial persists its RNG state +
    history line so a Ctrl+C'd run can be resumed by re-instantiating with
    ``resume=True``. The RNG state is saved AFTER the trial's resampling +
    evaluation completes, so a trial interrupted mid-evaluation re-uses the
    same RNG point on restart and re-draws the same config.
    """

    project: ProjectConfig
    storage_dir: Path | None = None
    resume: bool = False
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
        n_validation_rejects = 0
        prior_wall_s = 0.0

        history_path = RunLayout(base=self.storage_dir).history if self.storage_dir else None
        rng_state_path = self.storage_dir / _RNG_STATE_NAME if self.storage_dir else None
        wall_clock_path = self.storage_dir / _WALL_CLOCK_NAME if self.storage_dir else None

        if self.resume and history_path is not None and history_path.exists():
            history = _load_history(history_path)
            if rng_state_path is not None and rng_state_path.exists():
                try:
                    rng.setstate(pickle.loads(rng_state_path.read_bytes()))
                except Exception:
                    logger.warning(
                        "Could not restore RNG state from %s; falling back to a "
                        "fresh seeded RNG. Trials after resume may diverge from "
                        "what a single-process run would have drawn.",
                        rng_state_path,
                        exc_info=True,
                    )
            if wall_clock_path is not None and wall_clock_path.exists():
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
                "Resuming random search from trial %d/%d (prior wall=%.1fs)",
                len(history) + 1,
                budget.max_trials,
                prior_wall_s,
            )
        # Wiping stale state on a fresh start is the bench-level ``--clean``
        # flag's job (``_clear_output_root_for`` in run.py). We deliberately
        # do NOT wipe here, even when ``resume=False``: a user who passes
        # ``--no-clean`` without ``--resume`` keeps whatever was on disk.

        trial_usd_total = sum(h.eval_usd for h in history)

        t_start = time.monotonic()
        for trial_num in range(len(history) + 1, budget.max_trials + 1):
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

            if history_path is not None:
                _append_history(history_path, entry)
            if rng_state_path is not None:
                _atomic_pickle(rng_state_path, rng.getstate())
            if wall_clock_path is not None:
                cumulative = prior_wall_s + (time.monotonic() - t_start)
                wall_clock_path.write_text(json.dumps({"wall_clock_s": cumulative}), encoding="utf-8")

            best = max(h.answer_accuracy for h in history)
            logger.info(
                "random trial %d done | accuracy=%.3f | best so far=%.3f", trial_num, result.answer_accuracy, best
            )
            logger.info("")

        if not history:
            raise RuntimeError("Random search produced no successful trials")

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
            extras={"n_validation_rejects": n_validation_rejects},
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


def _atomic_pickle(path: Path, obj: object) -> None:
    """Pickle ``obj`` to ``path`` via a tmp + rename so a Ctrl+C during the
    write can't leave a half-written file that fails to unpickle on resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)

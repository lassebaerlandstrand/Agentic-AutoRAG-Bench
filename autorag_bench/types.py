"""Shared types: ``Optimizer`` protocol + dataclasses every method writes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_autorag.config.models import TrialConfig


@dataclass(frozen=True)
class Budget:
    """Bounds the optimizer's resource consumption.

    Sequential optimizers (Random / Bayesian / Agentic) honour ``max_trials``.
    AutoRAG enumerates per-node and ignores ``max_trials``; ``max_wall_clock_s``
    is the only knob that bounds it.
    """

    max_trials: int | None = None
    max_wall_clock_s: float | None = None


@dataclass(frozen=True)
class TrialResult:
    """One ``TrialConfig`` evaluated end-to-end by the framework's MCQ exam."""

    score: float
    metrics: dict[str, float]
    eval_usd: float

    @classmethod
    def from_exam_result(cls, exam_result: Any) -> TrialResult:
        """Build a ``TrialResult`` from the framework's ``ExamResult``."""
        return cls(
            score=float(exam_result.score),
            metrics={
                "answer_accuracy": float(exam_result.answer_accuracy),
                "mean_retrieval_quality": float(exam_result.mean_retrieval_quality),
                "mean_em": float(exam_result.mean_em),
                "mean_f1": float(exam_result.mean_f1),
            },
            eval_usd=float(getattr(exam_result, "total_llm_cost_usd", 0.0)),
        )


@dataclass
class HistoryEntry:
    """One trial in an optimizer's run history (sequential methods only)."""

    trial_number: int
    config: dict
    score: float
    metrics: dict[str, float]
    eval_usd: float

    def to_dict(self) -> dict:
        return {
            "trial_number": self.trial_number,
            "config": self.config,
            "score": self.score,
            "metrics": self.metrics,
            "eval_usd": self.eval_usd,
        }


@dataclass
class SearchResult:
    """The output every optimizer writes, regardless of search strategy.

    ``optimizer_usd`` and ``trial_usd_total`` are reported separately so the
    paper table can show whether a method "wins by spending more on the
    optimizer" (e.g. agentic reasoning) vs. "wins per trial of evaluation".
    """

    method: str
    seed: int | None
    deterministic: bool
    best_config: dict
    history: list[HistoryEntry]
    optimizer_usd: float
    trial_usd_total: float
    wall_clock_s: float
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "best_config": self.best_config,
            "history": [h.to_dict() for h in self.history],
            "optimizer_usd": self.optimizer_usd,
            "trial_usd_total": self.trial_usd_total,
            "wall_clock_s": self.wall_clock_s,
            "extras": self.extras,
        }


Evaluator = Callable[[TrialConfig], Awaitable[TrialResult]]
"""A trial evaluator: takes a config, returns a result.

Provided by the bench runner; wraps the framework's ``Orchestrator.evaluate_trial``.
"""


class Optimizer(Protocol):
    """Every method (random, bayesian, agentic, autorag) implements this."""

    name: str
    deterministic: bool

    async def search(
        self,
        evaluator: Evaluator,
        budget: Budget,
        *,
        seed: int | None = None,
    ) -> SearchResult: ...

"""Shared types: ``Optimizer`` protocol + dataclasses every method writes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_autorag.config.models import TrialConfig


@dataclass(frozen=True)
class Budget:
    """Bounds the optimizer's resource consumption.

    Sequential optimizers (Random / MO-TPE / Agentic) honour ``max_trials``.
    AutoRAG enumerates per-node and ignores ``max_trials``; ``max_wall_clock_s``
    is the only knob that bounds it.
    """

    max_trials: int | None = None
    max_wall_clock_s: float | None = None


@dataclass(frozen=True)
class TrialResult:
    """One ``TrialConfig`` evaluated end-to-end by the framework's open-ended exam.

    ``prompt_tokens`` / ``completion_tokens`` / ``embedding_tokens`` are summed
    over every cost-ledger bucket touched during this trial (rag_eval, judge,
    agent_proposal for agentic methods, embedding_build credit on first-use).
    Stay 0 for adapters that don't surface ledger totals.

    Per-trial accounting deliberately excludes pre-trial setup spend (exam
    generation, RAGAS QA generation, endpoint verification, probe-phase
    embedding builds). This is the bench's fairness rule: shared
    infrastructure cost is not attributed to any method's per-trial total.
    The framework's run-level ``cost_breakdown.json`` includes everything
    for audit.
    """

    answer_accuracy: float
    metrics: dict[str, float]
    eval_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    # Deploy-time LLM cost per query (synthesis + query-expansion), averaged
    # over valid questions. Read straight from ``ExamResult`` — it is the
    # SECOND objective the cost-aware optimizer (agentic_cost / motpe Pareto)
    # minimizes, and it deliberately EXCLUDES the judge + embedding-build
    # spend that ``eval_usd`` rolls in. Defaults to 0.0 for accuracy-only runs.
    mean_llm_cost_per_query_usd: float = 0.0

    @classmethod
    def from_exam_result(cls, exam_result: Any) -> TrialResult:
        """Build a ``TrialResult`` from the framework's ``ExamResult``."""
        return cls(
            answer_accuracy=float(exam_result.answer_accuracy),
            metrics={
                "answer_accuracy": float(exam_result.answer_accuracy),
                "mean_retrieval_quality": float(exam_result.mean_retrieval_quality),
                "mean_em": float(exam_result.mean_em),
                "mean_f1": float(exam_result.mean_f1),
            },
            eval_usd=float(getattr(exam_result, "total_llm_cost_usd", 0.0)),
            prompt_tokens=int(getattr(exam_result, "total_prompt_tokens", 0)),
            completion_tokens=int(getattr(exam_result, "total_completion_tokens", 0)),
            embedding_tokens=0,
            mean_llm_cost_per_query_usd=float(getattr(exam_result, "mean_llm_cost_per_query_usd", 0.0)),
        )


@dataclass
class HistoryEntry:
    """One trial in an optimizer's run history (sequential methods only)."""

    trial_number: int
    config: dict
    answer_accuracy: float
    metrics: dict[str, float]
    eval_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    # Per-query deploy-time LLM cost (the cost-aware Pareto objective). 0.0 on
    # accuracy-only runs; surfaced so the Pareto experiment can read it straight
    # off ``history.jsonl`` for every method, not just agentic_cost.
    mean_llm_cost_per_query_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trial_number": self.trial_number,
            "config": self.config,
            "answer_accuracy": self.answer_accuracy,
            "metrics": self.metrics,
            "eval_usd": self.eval_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "mean_llm_cost_per_query_usd": self.mean_llm_cost_per_query_usd,
        }


@dataclass
class SearchResult:
    """The output every optimizer writes, regardless of search strategy.

    ``optimizer_usd`` and ``trial_usd_total`` are reported separately so the
    paper table can show whether a method "wins by spending more on the
    optimizer" (e.g. agentic reasoning) vs. "wins per trial of evaluation".

    Token totals roll up across every trial in ``history`` plus any
    optimizer-side tokens (e.g. AutoRAG's enumeration LLM calls or the
    agentic proposer/diagnoser) that don't belong to a specific trial.
    """

    method: str
    seed: int | None
    deterministic: bool
    best_config: dict
    history: list[HistoryEntry]
    optimizer_usd: float
    trial_usd_total: float
    wall_clock_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
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
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "extras": self.extras,
        }


Evaluator = Callable[[TrialConfig], Awaitable[TrialResult]]
"""A trial evaluator: takes a config, returns a result.

Provided by the bench runner; wraps the framework's ``Orchestrator.evaluate_trial``.
"""


class Optimizer(Protocol):
    """Every method (random, motpe, agentic, autorag) implements this."""

    name: str
    deterministic: bool

    async def search(
        self,
        evaluator: Evaluator,
        budget: Budget,
        *,
        seed: int | None = None,
    ) -> SearchResult: ...

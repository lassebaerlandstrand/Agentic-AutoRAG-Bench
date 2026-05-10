"""Thin wrapper over the framework's HotpotQA prep + held-out scoring.

The bench runner reuses the framework's ``benchmark-prepare`` (called once per
matrix run; subsequent runs hit the cache) and ``benchmark-evaluate``
(per-winning-config). Wrapping both behind a small class keeps the run
orchestrator readable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from agentic_autorag.benchmark_eval.runner import run as run_benchmark_evaluate
from agentic_autorag.benchmarks import prepare as prepare_benchmark
from agentic_autorag.config.models import TrialConfig

logger = logging.getLogger("autorag_bench.run")


@dataclass
class HotpotQABenchmark:
    """Materialise corpus + qa.json once, then score every winning config against the held-out QA."""

    output_dir: Path
    split: str = "validation"
    sample_size: int | None = 2000
    seed: int = 42

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def corpus_dir(self) -> Path:
        return self.output_dir / "corpus"

    @property
    def qa_path(self) -> Path:
        return self.output_dir / "qa.json"

    def prepare(self) -> None:
        """Idempotent: skip prep when corpus + qa.json already exist."""
        if self.corpus_dir.is_dir() and self.qa_path.is_file():
            logger.info("HotpotQA corpus already prepared at %s", self.output_dir)
            return
        logger.info("Preparing HotpotQA (split=%s, sample_size=%s) at %s",
                    self.split, self.sample_size, self.output_dir)
        prepare_benchmark(
            name="hotpot_qa",
            output_dir=self.output_dir,
            split=self.split,
            sample_size=self.sample_size,
            seed=self.seed,
        )

    async def evaluate(
        self,
        *,
        project_config_path: str,
        trial_config: TrialConfig,
        output_path: Path,
        judge_model: str | None,
        limit: int | None = None,
        concurrency: int = 10,
    ) -> dict:
        """Score one winning ``TrialConfig`` against the held-out QA.

        Writes ``benchmark_results.json`` to ``output_path``. The framework's
        runner takes a path to the trial yaml; we materialise the trial config
        next to ``output_path`` for that interface and clean it up after.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trial_yaml_path = output_path.parent / "trial_for_eval.yaml"
        trial_yaml_path.write_text(
            yaml.safe_dump(trial_config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        try:
            result = await run_benchmark_evaluate(
                project_config_path=project_config_path,
                trial_config_path=str(trial_yaml_path),
                qa_path=str(self.qa_path),
                output_path=str(output_path),
                judge_model=judge_model,
                concurrency=concurrency,
                limit=limit,
            )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
        finally:
            trial_yaml_path.unlink(missing_ok=True)

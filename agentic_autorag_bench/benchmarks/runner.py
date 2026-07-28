"""Thin wrapper over the framework's benchmark prep + held-out scoring.

The bench runner reuses the framework's ``benchmark-prepare`` (called once per
matrix run; subsequent runs hit the cache) and ``benchmark-evaluate``
(per-winning-config). The benchmark name is passed through; the framework's
``ADAPTERS`` registry dispatches to the right adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from agentic_autorag.benchmark_eval.runner import run as run_benchmark_evaluate
from agentic_autorag.benchmarks import prepare as prepare_benchmark
from agentic_autorag.config.models import TrialConfig

logger = logging.getLogger("agentic_autorag_bench.run")


@dataclass
class BenchmarkRunner:
    """Materialise corpus + qa.json once, then score every winning config against the held-out QA."""

    name: str
    output_dir: Path
    split: str = "validation"
    sample_size: int | None = 2000
    seed: int = 42
    # HuggingFace dataset revision to pin. Left None the adapters resolve the
    # dataset's current head, which silently makes a rerun a different corpus
    # from the published one -- so the paper configs pin the sha recorded in
    # ``benchmark_data/<ds>/metadata.json``.
    hf_revision: str | None = None

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
            logger.info("%s corpus already prepared at %s", self.name, self.output_dir)
            return
        logger.info(
            "Preparing %s (split=%s, sample_size=%s, hf_revision=%s) at %s",
            self.name,
            self.split,
            self.sample_size,
            self.hf_revision or "<head>",
            self.output_dir,
        )
        prepare_benchmark(
            name=self.name,
            output_dir=self.output_dir,
            split=self.split,
            sample_size=self.sample_size,
            seed=self.seed,
            hf_revision=self.hf_revision,
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
        exclude_question_types: list[str] | None = None,
        qa_path_override: str | Path | None = None,
    ) -> dict:
        """Score one winning ``TrialConfig`` against the held-out QA.

        Writes ``benchmark_results.json`` to ``output_path``. The framework's
        runner takes a path to the trial yaml; we materialise the trial config
        next to ``output_path`` for that interface and clean it up after.

        ``exclude_question_types`` is forwarded to the framework runner as a
        general escape hatch for dropping a broken question type before scoring
        (unused by the paper configs; abstention rows are scored, not dropped).
        ``qa_path_override`` points the held-out scoring at a specific QA file
        (e.g. the stratified ``splits/holdout_qa.json``) instead of the full
        ``qa.json`` — the fix for the biased contiguous held-out slice.
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
                qa_path=str(qa_path_override or self.qa_path),
                output_path=str(output_path),
                judge_model=judge_model,
                concurrency=concurrency,
                limit=limit,
                exclude_question_types=exclude_question_types,
            )
            return result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
        finally:
            trial_yaml_path.unlink(missing_ok=True)

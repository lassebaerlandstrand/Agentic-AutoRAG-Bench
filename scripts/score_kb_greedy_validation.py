#!/usr/bin/env python
"""Score each ``kb_greedy`` config against the *validation* exam.

``run_kb_greedy`` evaluates the KB's strongest config only on the held-out gold,
so kb_greedy carries no validation-exam ``answer_accuracy`` and cannot appear on
the per-trial figures, which plot validation accuracy. This closes that gap: it
loads each ``kb_greedy/seed_<n>/best_config.yaml`` already on disk and scores it
through the same ``Orchestrator.evaluate_trial`` path, against the same
``examiner.custom_exam_path`` questions, with the same ``agent.judge_model`` the
searching methods use — so the number lands on their axis.

The score is written as a single-entry ``details/history.jsonl``
(``trial_number: 1``), which ``analyze.load_results`` already ingests.
``benchmark_results.json`` is never read or written here, so the held-out numbers
behind the paper's Table 1 stay frozen.

kb_greedy's config is byte-identical across seeds, but generation runs at
``temperature=1.0`` with no seed passed to the LLM, so each seed's evaluation is
an independent sample; that spread is the reference line's error band.

Run:  ``uv run python -u scripts/score_kb_greedy_validation.py``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import yaml
from agentic_autorag.config.models import TrialConfig
from agentic_autorag.orchestrator import Orchestrator
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench.types import HistoryEntry, TrialResult

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS = [
    REPO_ROOT / "configs" / "hotpot_paper.yaml",
    REPO_ROOT / "configs" / "musique_paper.yaml",
    REPO_ROOT / "configs" / "multihop_rag_paper.yaml",
]

logger = logging.getLogger("score_kb_greedy_validation")


def _seed_dirs(kb_greedy_root: Path, seeds: list[int] | None) -> list[Path]:
    """kb_greedy seed dirs holding a config to score, in numeric seed order."""
    dirs = [d for d in kb_greedy_root.glob("seed_*") if d.is_dir()]
    if seeds is not None:
        wanted = {f"seed_{s}" for s in seeds}
        dirs = [d for d in dirs if d.name in wanted]
    return sorted(dirs, key=lambda d: int(d.name.removeprefix("seed_")))


def _write_history(seed_dir: Path, trial: TrialConfig, exam_result: object) -> float:
    """Persist the validation score as a one-trial history; return the accuracy."""
    result = TrialResult.from_exam_result(exam_result)
    entry = HistoryEntry(
        trial_number=1,
        config=trial.model_dump(mode="json"),
        answer_accuracy=result.answer_accuracy,
        metrics=result.metrics,
        eval_usd=result.eval_usd,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        embedding_tokens=result.embedding_tokens,
        mean_llm_cost_per_query_usd=result.mean_llm_cost_per_query_usd,
    )
    layout = RunLayout(base=seed_dir)
    layout.ensure_details()
    layout.history.write_text(json.dumps(entry.to_dict()) + "\n", encoding="utf-8")
    return result.answer_accuracy


async def score_dataset(config_path: Path, seeds: list[int] | None, force: bool) -> None:
    """Score every kb_greedy seed of one dataset against its validation exam."""
    from agentic_autorag_bench.benchmarks.runner import BenchmarkRunner
    from agentic_autorag_bench.run import BenchConfig

    bench = BenchConfig.load(config_path)
    kb_greedy_root = bench.output_root / "kb_greedy"
    if not kb_greedy_root.is_dir():
        logger.warning("%s: no kb_greedy results at %s — skipping", config_path.name, kb_greedy_root)
        return

    pending = [
        d
        for d in _seed_dirs(kb_greedy_root, seeds)
        if (d / "best_config.yaml").exists() and (force or not RunLayout(base=d).history.exists())
    ]
    if not pending:
        logger.info("%s: every requested seed already scored — nothing to do", config_path.name)
        return

    # The corpus the orchestrator parses lives under the prepared benchmark dir;
    # prepare() is a no-op once the data is on disk.
    benchmark = BenchmarkRunner(
        name=bench.benchmark.name,
        output_dir=bench.benchmark.output_dir,
        split=bench.benchmark.split,
        sample_size=bench.benchmark.sample_size,
        seed=bench.benchmark.prep_seed,
    )
    benchmark.prepare()

    # One orchestrator per dataset: the corpus parse, validation-exam load, and
    # index build are shared across seeds, while each evaluate_trial call is a
    # fresh, independent evaluation.
    logger.info("%s: setting up orchestrator (loads examiner.custom_exam_path)", config_path.name)
    shared = Orchestrator(str(bench.project_config_path))
    shared.evaluator.quiet_per_question = True
    try:
        await shared.setup()
        for seed_dir in pending:
            trial = TrialConfig(**yaml.safe_load((seed_dir / "best_config.yaml").read_text(encoding="utf-8")))
            logger.info("%s: scoring %s on the validation exam", config_path.name, seed_dir.name)
            exam_result = await shared.evaluate_trial(trial)
            accuracy = _write_history(seed_dir, trial, exam_result)
            logger.info("%s: %s answer_accuracy=%.4f", config_path.name, seed_dir.name, accuracy)
    finally:
        await shared.cleanup()


async def main_async(configs: list[Path], seeds: list[int] | None, force: bool) -> None:
    from agentic_autorag.litellm_runtime import configure_litellm_runtime

    configure_litellm_runtime()
    for config_path in configs:
        logger.info("=" * 60)
        logger.info("DATASET %s", config_path.name)
        logger.info("=" * 60)
        await score_dataset(config_path, seeds, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--configs",
        type=lambda s: [Path(p).resolve() for p in s.split(",")],
        default=DEFAULT_CONFIGS,
        help="Comma-separated bench configs (default: the three Exp-1 paper configs)",
    )
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        help="Comma-separated seeds to score (default: every seed found on disk)",
    )
    parser.add_argument("--force", action="store_true", help="Rescore seeds that already have a history.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main_async(args.configs, args.seeds, args.force))


if __name__ == "__main__":
    main()

"""AutoRAG driver — orchestrates corpus export, QA generation, subprocess invoke, translation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from agentic_autorag.orchestrator import Orchestrator

from autorag_bench.methods.autorag.corpus_export import export_corpus_to_parquet
from autorag_bench.methods.autorag.native_config import generate_autorag_config
from autorag_bench.methods.autorag.qa_mcq import export_mcq_exam_to_parquet
from autorag_bench.methods.autorag.qa_ragas import export_ragas_qa_via_subprocess
from autorag_bench.methods.autorag.translator import translate_extracted_to_trial_config
from autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult, TrialResult

logger = logging.getLogger("autorag_bench.run")

QAVariant = Literal["ragas", "mcq"]


def _find_extracted_sample(project_dir: Path) -> Path | None:
    for candidate in sorted(project_dir.rglob("extracted_sample.yaml")):
        return candidate
    return None


def _find_latest_trial_dir(project_dir: Path) -> Path | None:
    """AutoRAG writes trial outputs as numbered subdirs under project_dir (0/, 1/, ...).

    The ``extract_best_config`` command needs the trial dir, not the project
    dir. Returns the highest-numbered trial dir, or None if no trials yet.
    """
    candidates: list[tuple[int, Path]] = []
    for child in project_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        candidates.append((int(child.name), child))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


@dataclass
class AutoRAGOptimizer:
    """Marker-Inc AutoRAG baseline (RAGAS-native or MCQ-ablation variant).

    AutoRAG runs in its own venv (path passed via ``autorag_python``). It does
    not consume the bench's ``evaluator`` callback — its evaluation loop is
    internal to AutoRAG. After it produces a winning pipeline we translate it
    back to a ``TrialConfig`` and re-score on the bench's evaluator so the
    ``best_config`` slot in the SearchResult has comparable metrics.
    """

    config_path: str
    output_dir: str
    qa_variant: QAVariant
    autorag_python: str | None = None
    name: str = ""
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"autorag_{self.qa_variant}"

    async def search(
        self,
        evaluator: Evaluator,
        budget: Budget,  # noqa: ARG002 — AutoRAG's enumeration ignores trial budgets
        *,
        seed: int | None = None,  # noqa: ARG002 — AutoRAG is deterministic conditional on inputs
    ) -> SearchResult:
        out_dir = Path(self.output_dir)
        autorag_dir = out_dir / "autorag_project"
        autorag_dir.mkdir(parents=True, exist_ok=True)

        autorag_python = self.autorag_python or os.environ.get("AUTORAG_PYTHON")
        if not autorag_python:
            raise RuntimeError(
                "AUTORAG_PYTHON not set. Run scripts/setup_autorag_venv.sh first or pass --autorag-python."
            )

        orch = Orchestrator(self.config_path, output_dir_override=str(out_dir))
        await orch.setup()

        t_start = time.monotonic()

        corpus_parquet = autorag_dir / "corpus.parquet"
        n_corpus = export_corpus_to_parquet(Path(orch.config.meta.corpus_path), corpus_parquet)
        logger.info("Exported %d documents to %s", n_corpus, corpus_parquet.name)

        qa_parquet = autorag_dir / "qa.parquet"
        if self.qa_variant == "mcq":
            exam_json = orch.cache_dir / "exam.json"
            if not exam_json.exists():
                raise RuntimeError(
                    f"AutoRAG-MCQ requires a cached exam.json at {exam_json}. "
                    "Run the agentic baseline first (or any baseline that triggers exam generation)."
                )
            n_qa = export_mcq_exam_to_parquet(exam_json, qa_parquet)
            logger.info("Exported %d MCQ rows to %s", n_qa, qa_parquet.name)
            shutil.copy2(Path(__file__).parent / "mcq_metric.py", autorag_dir / "mcq_metric.py")
        else:
            sample_n = orch.config.examiner.exam_size
            export_ragas_qa_via_subprocess(
                corpus_parquet,
                qa_parquet,
                sample_n=sample_n,
                llm_model=orch.config.agent.examiner_model,
                autorag_python=autorag_python,
            )

        autorag_config_dict, notes = generate_autorag_config(orch.config.search_space, qa_variant=self.qa_variant)
        autorag_config_path = autorag_dir / "autorag_config.yaml"
        autorag_config_path.write_text(yaml.safe_dump(autorag_config_dict, sort_keys=False), encoding="utf-8")
        (autorag_dir / "translation_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

        # AutoRAG installs an ``autorag`` console script next to ``python`` in
        # the venv; the package has no __main__, so ``python -m autorag`` won't
        # work. Locate the console script alongside the python interpreter.
        autorag_bin = Path(autorag_python).parent / "autorag"
        if not autorag_bin.exists():
            raise RuntimeError(
                f"AutoRAG console script not found at {autorag_bin}. "
                "Re-run scripts/setup_autorag_venv.sh."
            )

        env = dict(os.environ)

        # Skip the long subprocess if a previous invocation already produced
        # a trial dir AND its extracted_sample.yaml. Re-extract is cheap.
        existing_trial = _find_latest_trial_dir(autorag_dir)
        if existing_trial is None or not (existing_trial / "summary.csv").exists():
            logger.info("Invoking AutoRAG evaluate (%s variant)", self.qa_variant)
            result = subprocess.run(
                [
                    str(autorag_bin), "evaluate",
                    "--config", str(autorag_config_path),
                    "--qa_data_path", str(qa_parquet),
                    "--corpus_data_path", str(corpus_parquet),
                    "--project_dir", str(autorag_dir),
                ],
                check=False, env=env, capture_output=True, text=True,
            )
            if result.stdout:
                logger.info(result.stdout.rstrip())
            if result.stderr:
                logger.warning(result.stderr.rstrip())
            if result.returncode != 0:
                # Common failure paths worth surfacing:
                #   - Azure ResponsibleAIPolicyViolation (content filter):
                #     a single offending question kills the whole AutoRAG
                #     enumeration. Document it; the matrix runner catches
                #     this exception and the row is omitted from the table.
                stderr = result.stderr or ""
                if "ResponsibleAIPolicyViolation" in stderr or "content_filter" in stderr:
                    raise RuntimeError(
                        f"AutoRAG evaluate hit Azure content filter "
                        f"(rc={result.returncode}). Skipping this AutoRAG row; "
                        "see translation_notes.json for known limitations."
                    )
                raise RuntimeError(f"AutoRAG evaluate exited with rc={result.returncode}")

        # AutoRAG v0.3 doesn't auto-emit extracted_sample.yaml — the user must
        # call ``autorag extract_best_config`` explicitly with the trial dir.
        trial_dir = _find_latest_trial_dir(autorag_dir)
        if trial_dir is None:
            raise RuntimeError(f"AutoRAG produced no trial directory under {autorag_dir}")
        extracted = autorag_dir / "extracted_sample.yaml"
        if not extracted.exists():
            logger.info("Extracting best config from trial %s", trial_dir.name)
            result = subprocess.run(
                [
                    str(autorag_bin), "extract_best_config",
                    "--trial_path", str(trial_dir),
                    "--output_path", str(extracted),
                ],
                check=False, env=env, capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"AutoRAG extract_best_config exited with rc={result.returncode}: {result.stderr}"
                )
        if not extracted.exists():
            raise RuntimeError(f"AutoRAG produced no extracted_sample.yaml at {extracted}")

        trial_config = translate_extracted_to_trial_config(extracted, orch.config.search_space)
        violations = orch.config.validate_trial(trial_config)
        if violations:
            logger.warning("Translated config has validation issues (saving anyway): %s", "; ".join(violations))

        # Re-score the winning translated config on the bench evaluator so the
        # ``best_config``'s metrics are directly comparable to other rows.
        rescore: TrialResult = await evaluator(trial_config)

        include_graph = orch.config.uses_graph()
        trial_dump = trial_config.to_prompt_dump(include_graph=include_graph)
        history = [
            HistoryEntry(
                trial_number=1,
                config=trial_dump,
                score=rescore.score,
                metrics=rescore.metrics,
                eval_usd=rescore.eval_usd,
            )
        ]
        wall_clock = time.monotonic() - t_start

        return SearchResult(
            method=self.name,
            seed=None,
            deterministic=self.deterministic,
            best_config=trial_dump,
            history=history,
            optimizer_usd=0.0,  # AutoRAG's internal eval cost lives in extras.subprocess_cost
            trial_usd_total=rescore.eval_usd,
            wall_clock_s=wall_clock,
            extras={
                "qa_variant": self.qa_variant,
                "translation_notes": notes,
                "extracted_sample_path": str(extracted),
                "autorag_python": autorag_python,
            },
        )

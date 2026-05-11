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

from agentic_autorag_bench.methods.autorag.corpus_export import export_corpus_to_parquet
from agentic_autorag_bench.methods.autorag.native_config import generate_autorag_config
from agentic_autorag_bench.methods.autorag.qa_mcq import export_mcq_exam_to_parquet
from agentic_autorag_bench.methods.autorag.qa_prescreen import prescreen_qa_for_content_filter
from agentic_autorag_bench.methods.autorag.qa_ragas import export_ragas_qa_via_subprocess
from agentic_autorag_bench.methods.autorag.translator import translate_extracted_to_trial_config
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult, TrialResult

logger = logging.getLogger("agentic_autorag_bench.run")

QAVariant = Literal["ragas", "mcq"]


def _find_extracted_sample(project_dir: Path) -> Path | None:
    for candidate in sorted(project_dir.rglob("extracted_sample.yaml")):
        return candidate
    return None


# Env vars whose literal values must never land in a committed YAML / JSON.
# When AutoRAG's ``extract_best_config`` expands ``${VAR}`` placeholders we
# undo the expansion so the artifact is safe to commit. Ordered so longer
# prefixes are matched first (e.g. AZURE_AI_API_KEY before AZURE_API_KEY).
_SCRUB_ENV_VARS = (
    "AZURE_AI_API_KEY",
    "AZURE_AI_API_BASE",
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "OPENAI_API_KEY",
)


def _scrub_env_placeholders(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    scrubbed = text
    for var in _SCRUB_ENV_VARS:
        literal = os.environ.get(var)
        if literal and literal in scrubbed:
            scrubbed = scrubbed.replace(literal, "${" + var + "}")
    if scrubbed != text:
        path.write_text(scrubbed, encoding="utf-8")
        logger.info("Scrubbed env-var placeholders in %s", path.name)


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

        # Drop rows that Azure's content filter rejects before AutoRAG's
        # enumerate subprocess sees them — even one rejection aborts the
        # whole AutoRAG run, so this is the only place we get to intervene.
        # Probe with the cheapest search-space LLM since Azure's filter
        # applies at the gateway, not per-deployment.
        prescreen_model = orch.config.search_space.llm_models[0]
        dropped = await prescreen_qa_for_content_filter(qa_parquet, model=prescreen_model)
        if dropped:
            notes = dict(notes)
            notes["content_filter_dropped_qids"] = dropped
            notes["content_filter_prescreen_model"] = prescreen_model

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

        # SECURITY: AutoRAG's ``extract_best_config`` substitutes ``${AZURE_API_KEY}``
        # (and any other env-var placeholders) with their literal values before
        # writing the file. This file is part of the committed paper artifact,
        # so the literal key would leak into git. Scrub it back to the placeholder
        # form before anything else reads or persists it.
        _scrub_env_placeholders(extracted)

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
            # ``optimizer_usd`` is set to 0 deliberately: AutoRAG's enumerate
            # subprocess makes many internal LLM calls during pipeline
            # evaluation (one per (config × QA-row) for each generator module)
            # but does not surface a token-level cost ledger. AutoRAG only
            # records ``average_output_token`` per module in summary.csv —
            # prompt tokens, the dominant share of cost, are not captured.
            # We therefore *cannot* report a comparable optimizer_usd here
            # without re-tokenising every prompt, which would conflate this
            # with the real cost we'd see in production. The ``trial_usd_total``
            # below reflects only the bench-side re-scoring of the winning
            # config (apples-to-apples with the other rows' final-trial cost).
            # Reviewers should treat the cost column as a strict lower bound;
            # the ``cost_caveat`` flag in ``extras`` is consumed by analyze.py
            # to surface the disclaimer in Table_1.md and figure_efficiency.png.
            optimizer_usd=0.0,
            trial_usd_total=rescore.eval_usd,
            wall_clock_s=wall_clock,
            extras={
                "qa_variant": self.qa_variant,
                "translation_notes": notes,
                "extracted_sample_path": str(extracted),
                "autorag_python": autorag_python,
                "cost_caveat": (
                    "AutoRAG's internal enumeration cost is not instrumented; "
                    "optimizer_usd=0 and trial_usd_total reflects only the "
                    "bench-side re-scoring of the winning config. The true "
                    "search cost is higher."
                ),
            },
        )

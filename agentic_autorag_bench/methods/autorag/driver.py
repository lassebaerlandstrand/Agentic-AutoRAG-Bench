"""AutoRAG driver — orchestrates corpus export, QA generation, subprocess invoke, translation."""

from __future__ import annotations

import hashlib
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
from agentic_autorag_bench.methods.autorag.qa_our_exam import export_open_exam_to_parquet
from agentic_autorag_bench.methods.autorag.qa_prescreen import prescreen_qa_for_content_filter
from agentic_autorag_bench.methods.autorag.qa_ragas import export_ragas_qa_via_subprocess
from agentic_autorag_bench.methods.autorag.translator import translate_extracted_to_trial_config
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult, TrialResult

logger = logging.getLogger("agentic_autorag_bench.run")

QAVariant = Literal["ragas", "our_exam"]


def _find_extracted_sample(project_dir: Path) -> Path | None:
    for candidate in sorted(project_dir.rglob("extracted_sample.yaml")):
        return candidate
    return None


# Convention: scripts/setup_autorag_venv.sh creates .autorag-venv at the bench
# repo root. This driver lives at agentic_autorag_bench/methods/autorag/, so
# the bench root is three parents up from this file's directory.
_BENCH_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AUTORAG_VENV_PYTHON = _BENCH_ROOT / ".autorag-venv" / "bin" / "python"


def resolve_autorag_python(explicit: str | None = None) -> str:
    """Resolve the path to the AutoRAG venv's python interpreter.

    Priority: ``explicit`` arg > ``AUTORAG_PYTHON`` env var > the conventional
    ``<bench_root>/.autorag-venv/bin/python`` path produced by
    ``scripts/setup_autorag_venv.sh``. Raises with setup instructions if none
    resolve to a usable interpreter + ``autorag`` console script.

    Also used by ``run.py`` as a fail-fast preflight: a whole 5-method matrix
    would otherwise spend hours on agentic/random/bayesian before the AutoRAG
    rows discover the missing interpreter at ``search()`` time.
    """
    candidate = explicit or os.environ.get("AUTORAG_PYTHON") or str(_DEFAULT_AUTORAG_VENV_PYTHON)
    if not Path(candidate).exists():
        raise RuntimeError(
            f"AutoRAG interpreter not found at {candidate}. "
            "Run scripts/setup_autorag_venv.sh first, "
            "or set AUTORAG_PYTHON to a custom path."
        )
    # AutoRAG installs an ``autorag`` console script next to ``python`` in the
    # venv; the package has no __main__, so ``python -m autorag`` won't work.
    autorag_bin = Path(candidate).parent / "autorag"
    if not autorag_bin.exists():
        raise RuntimeError(
            f"AutoRAG console script not found at {autorag_bin}. "
            "Re-run scripts/setup_autorag_venv.sh."
        )
    return candidate


# Env vars whose literal values must never land in a committed YAML / JSON.
# When AutoRAG's ``extract_best_config`` expands ``${VAR}`` placeholders we
# undo the expansion so the artifact is safe to commit. Ordered so longer
# prefixes are matched first (e.g. AZURE_AI_API_KEY before AZURE_API_KEY).
# AWS_REGION_NAME is included because native_config.py embeds it as
# ``${AWS_REGION_NAME}`` for the bedrock_converse modules; the literal region
# isn't a secret but we keep the placeholder form for environment portability.
_SCRUB_ENV_VARS = (
    "AZURE_AI_API_KEY",
    "AZURE_AI_API_BASE",
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION_NAME",
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


# AutoRAG materialises ``${AZURE_API_KEY}`` etc. into its per-trial config.yaml
# and every node's summary.csv ``module_params`` column. We sweep both
# extensions after the evaluate subprocess so committed artifacts never carry
# live credentials. Scope is the trial dir only — we don't touch parquet
# (binary, no env-var expansion path) or the cache dir (symlinked out).
_SCRUB_GLOBS = ("*.yaml", "*.yml", "*.csv", "*.json")


def _scrub_trial_artifacts(trial_dir: Path) -> None:
    for pattern in _SCRUB_GLOBS:
        for path in trial_dir.rglob(pattern):
            _scrub_env_placeholders(path)


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


def _summarize_cost_log(log_path: Path) -> dict:
    """Sum the per-call log into a USD total + per-model + per-source breakdown.

    ``cost_tracker.py`` writes one JSONL line per LLM completion. Pricing
    comes from ``litellm.cost_per_token`` and includes cache discounts when
    the call recorded ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
    (OpenAI's implicit cache, Bedrock cachePoint). Calls for models LiteLLM
    doesn't price (e.g. local self-hosted endpoints, exotic Bedrock IDs)
    contribute zero.

    Returns ``{total_usd, buckets (by model), by_source (by source), n_calls}``.
    ``by_source`` lets the driver subtract one-time setup spend
    (``source="qa_ragas"`` — RAGAS QA generation, the autorag_ragas analogue
    of our exam generation) from the headline tally without losing it from
    ``cost_breakdown.json``.
    """
    import litellm

    buckets: dict[str, dict] = {}
    by_source: dict[str, dict] = {}
    total_usd = 0.0
    total_calls = 0
    if not log_path.exists():
        return {"total_usd": 0.0, "buckets": {}, "by_source": {}, "n_calls": 0}

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed cost log line: %s", line[:200])
                continue
            model = rec.get("model") or "unknown"
            # Default to "autorag_eval" so older log lines (pre-source-field)
            # don't accidentally land in qa_ragas's bucket and get excluded.
            source = rec.get("source") or "autorag_eval"
            prompt_tokens = int(rec.get("prompt_tokens", 0) or 0)
            completion_tokens = int(rec.get("completion_tokens", 0) or 0)
            cache_read = int(rec.get("cache_read_input_tokens", 0) or 0)
            cache_creation = int(rec.get("cache_creation_input_tokens", 0) or 0)
            try:
                in_usd, out_usd = litellm.cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_read_input_tokens=cache_read,
                    cache_creation_input_tokens=cache_creation,
                )
                call_usd = float(in_usd or 0.0) + float(out_usd or 0.0)
            except Exception:
                logger.debug("No litellm pricing for model=%s", model, exc_info=True)
                call_usd = 0.0
            for target_key, target_map in (("by_model", buckets), ("by_source", by_source)):
                key = model if target_key == "by_model" else source
                bucket = target_map.setdefault(
                    key,
                    {
                        "usd": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "n_calls": 0,
                    },
                )
                bucket["usd"] += call_usd
                bucket["prompt_tokens"] += prompt_tokens
                bucket["completion_tokens"] += completion_tokens
                bucket["cache_read_input_tokens"] += cache_read
                bucket["cache_creation_input_tokens"] += cache_creation
                bucket["n_calls"] += 1
            total_usd += call_usd
            total_calls += 1

    return {"total_usd": total_usd, "buckets": buckets, "by_source": by_source, "n_calls": total_calls}


def _compute_resources_cache_key(corpus_parquet: Path, embedding_models: list[str]) -> str:
    """Stable hash over (corpus content, embedder set) for the AutoRAG cache.

    Both inputs determine what lives under ``resources/``: corpus content is
    embedded into Chroma + BM25 indexes; embedder names + ordering decide
    which Chroma collection (``embed_N``) holds which model's vectors. Adding
    or removing an embedder shifts collection indices and invalidates the
    whole cache, so we hash on both axes. Other axes that *don't* affect the
    on-disk artifacts (LLM choice, reranker, top_k) are excluded.
    """
    h = hashlib.sha256()
    with open(corpus_parquet, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    h.update(b"\0embedders\0")
    for m in embedding_models:
        h.update(m.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _setup_resources_cache(autorag_dir: Path, cache_root: Path, cache_key: str) -> Path:
    """Symlink ``<autorag_dir>/resources`` to a shared, corpus-keyed dir.

    AutoRAG writes Chroma collections + BM25 pickles + ``vectordb.yaml`` under
    ``<project_dir>/resources/`` (hard-coded in ``evaluator.py``). By making
    that subdir a symlink to a shared cache, ``filter_exist_ids`` and
    ``bm25_ingest`` see existing entries on subsequent runs and short-circuit
    the (slow) embedding phase. The cache key encodes corpus content + the
    embedder list, so changes to either invalidate cleanly.

    Returns the resolved cache directory path for logging.
    """
    cache_dir = cache_root / f"autorag_resources_{cache_key}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    resources_link = autorag_dir / "resources"
    desired_target = cache_dir.resolve()
    if resources_link.is_symlink():
        if Path(os.readlink(resources_link)).resolve() == desired_target:
            return cache_dir
        resources_link.unlink()
    elif resources_link.exists():
        # Previous run created a real directory (e.g. before caching landed).
        # Migrate any populated artifacts into the cache so we don't redo work,
        # then replace the dir with the symlink.
        for child in resources_link.iterdir():
            target = cache_dir / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
        shutil.rmtree(resources_link)
    resources_link.symlink_to(desired_target, target_is_directory=True)
    return cache_dir


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

        autorag_python = resolve_autorag_python(self.autorag_python)

        orch = Orchestrator(self.config_path, output_dir_override=str(out_dir))
        await orch.setup()

        t_start = time.monotonic()

        corpus_parquet = autorag_dir / "corpus.parquet"
        n_corpus = export_corpus_to_parquet(Path(orch.config.meta.corpus_path), corpus_parquet)
        logger.info("Exported %d documents to %s", n_corpus, corpus_parquet.name)

        # Symlink ``autorag_project/resources/`` to a corpus-keyed shared cache
        # so repeat runs reuse Chroma collections + BM25 pickles instead of
        # rebuilding from scratch. AutoRAG's own filter_exist_ids /
        # bm25_ingest handle the "already populated" case natively, so this
        # is plug-and-play: the first run pays the embedding cost; subsequent
        # runs skip straight to retrieval. Cache invalidates on (corpus,
        # embedder list) changes.
        cache_key = _compute_resources_cache_key(corpus_parquet, list(orch.config.search_space.embedding.models))
        cache_dir = _setup_resources_cache(autorag_dir, orch.cache_dir, cache_key)
        logger.info("AutoRAG resources cache: %s", cache_dir)

        qa_parquet = autorag_dir / "qa.parquet"
        # Both the qa_ragas QA-generation subprocess (when qa_variant=='ragas')
        # and the autorag-eval subprocess append to the same llm_calls.jsonl,
        # so ``_summarize_cost_log`` ends up reporting a single total for the
        # whole method run that includes the RAGAS QA generation step.
        cost_log_path = autorag_dir / "llm_calls.jsonl"
        if self.qa_variant == "our_exam":
            exam_json = orch.cache_dir / "exam.json"
            if not exam_json.exists():
                raise RuntimeError(
                    f"autorag_our_exam requires a cached exam.json at {exam_json}. "
                    "Run the agentic baseline first (or any baseline that triggers exam generation)."
                )
            n_qa = export_open_exam_to_parquet(exam_json, qa_parquet)
            logger.info("Exported %d exam rows to %s", n_qa, qa_parquet.name)
        else:
            sample_n = orch.config.examiner.exam_size
            export_ragas_qa_via_subprocess(
                corpus_parquet,
                qa_parquet,
                sample_n=sample_n,
                llm_model=orch.config.agent.examiner_model,
                autorag_python=autorag_python,
                cost_log_path=cost_log_path,
            )

        autorag_config_dict, notes = generate_autorag_config(orch.config.search_space, qa_variant=self.qa_variant)
        autorag_config_path = autorag_dir / "autorag_config.yaml"
        autorag_config_path.write_text(yaml.safe_dump(autorag_config_dict, sort_keys=False), encoding="utf-8")

        # Fail fast on missing bedrock env. AutoRAG's ``convert_env_in_dict``
        # silently substitutes unset ``${VAR}`` with empty string, so a
        # missing AWS_REGION_NAME doesn't error until boto3 tries to resolve
        # the bedrock endpoint mid-eval and raises NoRegionError after every
        # other node has already run. Surface it before the long subprocess.
        if notes.get("bedrock_in_search_space") and not os.environ.get("AWS_REGION_NAME"):
            raise RuntimeError(
                "AWS_REGION_NAME is unset but the search space contains bedrock/* models. "
                "AutoRAG's bedrock_converse LLM needs a region (boto3 doesn't read "
                "AWS_REGION_NAME on its own, so we pass it explicitly via ${AWS_REGION_NAME}). "
                "Either export AWS_REGION_NAME (e.g. us-east-1) or drop bedrock/* from the search-space stage pools."
            )

        # Drop rows that Azure's content filter rejects before AutoRAG's
        # enumerate subprocess sees them — even one rejection aborts the
        # whole AutoRAG run, so this is the only place we get to intervene.
        # Probe with the cheapest search-space LLM since Azure's filter
        # applies at the gateway, not per-deployment.
        prescreen_model = orch.config.search_space.generator.models[0]
        dropped = await prescreen_qa_for_content_filter(qa_parquet, model=prescreen_model)
        if dropped:
            notes = dict(notes)
            notes["content_filter_dropped_qids"] = dropped
            notes["content_filter_prescreen_model"] = prescreen_model

        (autorag_dir / "translation_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

        # resolve_autorag_python already verified the ``autorag`` console
        # script exists alongside the interpreter.
        autorag_bin = Path(autorag_python).parent / "autorag"
        cost_tracker_script = Path(__file__).parent / "cost_tracker.py"

        env = dict(os.environ)
        env["AUTORAG_COST_LOG"] = str(cost_log_path)

        # Skip the long subprocess if a previous invocation already produced
        # a trial dir AND its extracted_sample.yaml. Re-extract is cheap.
        existing_trial = _find_latest_trial_dir(autorag_dir)
        if existing_trial is None or not (existing_trial / "summary.csv").exists():
            # Fresh run: drop any prior cost log so totals reflect only this
            # invocation. (``--no-clean`` cases that *do* take the skip branch
            # below intentionally keep the old log so the cost matches the
            # trial dir we're reusing.)
            cost_log_path.unlink(missing_ok=True)
            logger.info("Invoking AutoRAG evaluate (%s variant)", self.qa_variant)
            # Route through cost_tracker.py instead of the bare autorag binary
            # so every OpenAI / Bedrock call's token usage is logged. RAGAS QA
            # generation runs in a separate subprocess (qa_ragas.py) that is
            # NOT wrapped, so exam-creation cost stays out of the meter — fair
            # across methods that reuse the agentic exam.
            result = subprocess.run(
                [
                    autorag_python, str(cost_tracker_script),
                    "evaluate",
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

        # SECURITY: AutoRAG's evaluate subprocess materialises ``${AZURE_API_KEY}``
        # (and any other env-var placeholders we feed it) into the per-trial
        # config.yaml and every node's summary.csv ``module_params`` column.
        # Scrub them back to placeholder form before any downstream step reads
        # the files or the bench commits them. Idempotent on re-runs, so it's
        # safe to call on the ``--no-clean`` skip path too.
        _scrub_trial_artifacts(trial_dir)
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
                prompt_tokens=rescore.prompt_tokens,
                completion_tokens=rescore.completion_tokens,
                embedding_tokens=rescore.embedding_tokens,
            )
        ]
        wall_clock = time.monotonic() - t_start

        # Aggregate the per-call cost log written by cost_tracker.py (eval
        # subprocess) and qa_ragas.py (when qa_variant=='ragas' — both append
        # to the same llm_calls.jsonl). Persist the full breakdown alongside
        # the trial dir for audit.
        cost_summary = _summarize_cost_log(cost_log_path)
        (autorag_dir / "cost_breakdown.json").write_text(
            json.dumps(cost_summary, indent=2), encoding="utf-8"
        )
        # Per-trial accounting fairness: RAGAS QA-generation is autorag_ragas's
        # one-time setup spend (its analogue of our exam generation), so the
        # bench excludes it from the headline tally — same rule that hides
        # exam-gen costs from the agentic side. Pull totals from non-qa_ragas
        # sources only; the full picture is still in cost_breakdown.json.
        bench_visible_sources = {
            src: bucket for src, bucket in cost_summary["by_source"].items() if src != "qa_ragas"
        }
        optimizer_usd = sum(float(b.get("usd", 0.0)) for b in bench_visible_sources.values())
        optimizer_prompt_tokens = sum(int(b.get("prompt_tokens", 0)) for b in bench_visible_sources.values())
        optimizer_completion_tokens = sum(int(b.get("completion_tokens", 0)) for b in bench_visible_sources.values())

        return SearchResult(
            method=self.name,
            seed=None,
            deterministic=self.deterministic,
            best_config=trial_dump,
            history=history,
            # AutoRAG's internal enumeration spend, headline-fair: excludes
            # the bench-side rescoring (in ``trial_usd_total``) AND the RAGAS
            # QA-generation subprocess (excluded for fairness — see the
            # ``bench_visible_sources`` filter above). The full picture
            # (including qa_ragas) is in ``autorag_dir/cost_breakdown.json``.
            optimizer_usd=optimizer_usd,
            trial_usd_total=rescore.eval_usd,
            wall_clock_s=wall_clock,
            prompt_tokens=optimizer_prompt_tokens + rescore.prompt_tokens,
            completion_tokens=optimizer_completion_tokens + rescore.completion_tokens,
            embedding_tokens=rescore.embedding_tokens,
            extras={
                "qa_variant": self.qa_variant,
                "translation_notes": notes,
                "extracted_sample_path": str(extracted),
                "autorag_python": autorag_python,
                "cost_breakdown_path": str(autorag_dir / "cost_breakdown.json"),
                "n_llm_calls": cost_summary["n_calls"],
            },
        )

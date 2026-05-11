"""Cross-method content-filter exclusion for hold-out scoring.

When any method's best config rejects a hold-out question due to a provider
content-policy filter, that question is dropped from *every* method's score
denominator. This keeps the paper's apples-to-apples comparison: all rows
score the same questions.

Pipeline:
1. After every (method, seed) hold-out completes, scan its ``benchmark_results.json``
   for per-question rows tagged ``CONTENT_FILTER``.
2. Union the ids across all (method, seed) outputs.
3. Rewrite each ``benchmark_results.json``: recompute aggregate fields with
   the union dropped from both numerator and denominator; tag the file with
   ``excluded_question_ids``.
4. Write ``filtered_questions.json`` at the run root summarising the union.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agentic_autorag.examiner._errors import CONTENT_FILTER_SENTINEL

logger = logging.getLogger("agentic_autorag_bench.run")

_RETRIEVAL_KS: tuple[int, ...] = (1, 2, 5, 10)


def _retrieval_metrics_for_row(row: dict) -> tuple[dict[int, float], int | None]:
    """Recompute recall@k and first-gold rank from a per_question row.

    Mirrors ``benchmark_eval.scoring.retrieval_metrics`` but doesn't import it
    to keep this module free of framework engine deps. The ``retrieved_doc_ids``
    list is already dedup-preserved in the framework; we don't re-dedupe here
    because the bench treats the persisted list as canonical.
    """
    supporting = row.get("supporting_doc_ids") or []
    retrieved = row.get("retrieved_doc_ids") or []
    if not supporting:
        return {k: 0.0 for k in _RETRIEVAL_KS}, None

    gold = set(supporting)
    seen: set[str] = set()
    dedup: list[str] = []
    for d in retrieved:
        if d in seen:
            continue
        seen.add(d)
        dedup.append(d)

    first_rank: int | None = None
    for rank, d in enumerate(dedup, start=1):
        if d in gold:
            first_rank = rank
            break

    recalls = {k: sum(1 for d in dedup[:k] if d in gold) / len(gold) for k in _RETRIEVAL_KS}
    return recalls, first_rank


def _collect_filtered_ids(method_files: list[Path]) -> tuple[set[str], dict[str, list[str]]]:
    """Scan every per_question for CONTENT_FILTER rows.

    Returns ``(union_ids, by_run)`` where ``by_run`` keys are "method/seed_dir"
    strings derived from the file path, so the registry can attribute each id
    to the run(s) that observed it.
    """
    union: set[str] = set()
    by_run: dict[str, list[str]] = {}
    for f in method_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        per_q = data.get("per_question", [])
        run_key = f"{f.parent.parent.name}/{f.parent.name}"
        ids = sorted(
            r["id"] for r in per_q if r.get("error") == CONTENT_FILTER_SENTINEL
        )
        if ids:
            by_run[run_key] = ids
            union.update(ids)
    return union, by_run


def _rescore_one(data: dict, excluded: set[str]) -> dict:
    """Return a new benchmark_results dict with aggregates excluding ``excluded``.

    Mirrors ``benchmark_eval.runner._aggregate`` but operates on the persisted
    per_question dicts and applies the union-exclusion before tallying. Any
    row in ``excluded`` is dropped from both numerator and denominator (judge
    accuracy, em, f1, retrieval) — same treatment whether or not *this* method
    tagged it as content-filtered, which is the whole point of the union.
    """
    per_q: list[dict] = data.get("per_question", [])
    error_sentinels = {"CONTENT_FILTER", "PERMANENT_LLM_ERROR", "TRANSIENT_LLM_ERROR"}
    valid = [
        r for r in per_q
        if r.get("id") not in excluded
        and r.get("error") not in error_sentinels
    ]
    n_valid = len(valid)
    judge_enabled = data.get("judge_model") is not None

    out = dict(data)
    out["excluded_question_ids"] = sorted(excluded)
    out["n_total"] = len(per_q) - sum(1 for r in per_q if r.get("id") in excluded)
    out["n_valid"] = n_valid

    if not n_valid:
        out.update({
            "em": 0.0,
            "f1": 0.0,
            "llm_judge_accuracy": None,
            "n_judge_invalid": 0,
            "recall_at_1": None,
            "recall_at_2": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "avg_retrieval_s": 0.0,
            "avg_generation_s": 0.0,
        })
        return out

    out["em"] = sum(float(r.get("em", 0.0)) for r in valid) / n_valid
    out["f1"] = sum(float(r.get("f1", 0.0)) for r in valid) / n_valid
    out["avg_retrieval_s"] = sum(float(r.get("retrieval_s", 0.0)) for r in valid) / n_valid
    out["avg_generation_s"] = sum(float(r.get("generation_s", 0.0)) for r in valid) / n_valid

    judged = [r for r in valid if r.get("judge") is not None]
    out["n_judge_invalid"] = (
        sum(1 for r in valid if r.get("judge") is None) if judge_enabled else 0
    )
    out["llm_judge_accuracy"] = (
        sum(int(r["judge"]) for r in judged) / len(judged)
        if judge_enabled and judged
        else None
    )

    supporting_present = any(r.get("supporting_doc_ids") for r in valid)
    if supporting_present:
        recall_sums = {k: 0.0 for k in _RETRIEVAL_KS}
        mrr_sum = 0.0
        n_with_gold = 0
        for r in valid:
            if not r.get("supporting_doc_ids"):
                continue
            n_with_gold += 1
            recalls, rank = _retrieval_metrics_for_row(r)
            for k in recall_sums:
                recall_sums[k] += recalls[k]
            mrr_sum += 1.0 / rank if rank else 0.0
        if n_with_gold:
            for k in recall_sums:
                out[f"recall_at_{k}"] = recall_sums[k] / n_with_gold
            out["mrr"] = mrr_sum / n_with_gold
        else:
            for k in _RETRIEVAL_KS:
                out[f"recall_at_{k}"] = None
            out["mrr"] = None
    else:
        for k in _RETRIEVAL_KS:
            out[f"recall_at_{k}"] = None
        out["mrr"] = None

    return out


def apply_union_exclusion(output_root: Path) -> dict:
    """Apply union content-filter exclusion across every (method, seed) hold-out.

    Walks ``output_root/<method>/<seed>/benchmark_results.json``, builds the
    union of CONTENT_FILTER ids, rewrites each json with rescored aggregates,
    and writes ``output_root/filtered_questions.json``. Returns a summary dict.
    """
    output_root = Path(output_root)
    method_files = sorted(output_root.glob("*/seed_*/benchmark_results.json"))
    method_files += sorted(output_root.glob("*/default/benchmark_results.json"))
    if not method_files:
        logger.info("No benchmark_results.json files found under %s — skipping union exclusion", output_root)
        return {"excluded_ids": [], "by_run": {}}

    union, by_run = _collect_filtered_ids(method_files)
    if union:
        logger.info(
            "Union-excluding %d content-filtered question(s) across %d hold-out run(s); "
            "contributing runs: %s",
            len(union), len(method_files), sorted(by_run.keys()),
        )
    else:
        logger.info(
            "No content-filter rejections across %d hold-out runs — "
            "tagging each results file with empty excluded_question_ids",
            len(method_files),
        )

    # Always rewrite so every benchmark_results.json has the same shape:
    # explicit ``excluded_question_ids`` field and aggregates recomputed from
    # the persisted per_question rows. Idempotent.
    for f in method_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        adjusted = _rescore_one(data, union)
        f.write_text(json.dumps(adjusted, indent=2), encoding="utf-8")

    registry = {
        "excluded_question_ids": sorted(union),
        "by_run": by_run,
        "n_runs_scanned": len(method_files),
    }
    (output_root / "filtered_questions.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )
    return registry

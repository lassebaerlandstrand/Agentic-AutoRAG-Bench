"""Deterministic stratified holdout split for benchmark QA.

A benchmark's ``qa.json`` is one flat pool. A held-out slice for final gold
scoring is drawn from it, **stratified** by the dataset's difficulty key
(MuSiQue ``n_hops``, HotpotQA ``type``, MultiHop-RAG ``question_type``) so it
mirrors the pool's difficulty mix — replacing the contiguous ``qa_pairs[:limit]``
held-out, which on a hop-sorted pool silently returned only the easiest stratum.
Everything else usable becomes the optimization reservoir the real-QA exam draws
its questions from. Rows with no gold documents (empty ``supporting_doc_ids``)
are excluded: they can't be grounded into an exam nor scored for retrieval.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from agentic_autorag.benchmarks.schema import BenchmarkQAPair
from pydantic import BaseModel

# The held-out slice size is the paper default; overridable at the call site.
# The optimization slice is everything else usable (a reservoir the real-QA
# builder draws from lazily), so there is no oversample size to tune.
DEFAULT_HOLDOUT_SIZE = 300

# Difficulty key per benchmark, tried in this order when none is given. Each
# names a ``metadata`` field whose value labels the stratum.
STRATIFY_KEY_PRIORITY: tuple[str, ...] = ("n_hops", "type", "question_type")


class SplitProvenance(BaseModel):
    """Reproducibility record written alongside the split files."""

    stratify_key: str
    seed: int
    n_pool_total: int
    n_excluded_no_docs: int
    n_pool_usable: int
    holdout_size: int
    opt_size: int
    pool_distribution: dict[str, int]
    holdout_distribution: dict[str, int]
    opt_distribution: dict[str, int]
    disjoint: bool


class SplitResult(BaseModel):
    holdout: list[BenchmarkQAPair]
    optimization: list[BenchmarkQAPair]
    provenance: SplitProvenance


def detect_stratify_key(pairs: list[BenchmarkQAPair]) -> str:
    """Pick the difficulty key present in the pool's metadata."""
    keys = set()
    for p in pairs:
        keys.update((p.metadata or {}).keys())
    for candidate in STRATIFY_KEY_PRIORITY:
        if candidate in keys:
            return candidate
    raise ValueError(
        f"no known stratify key in metadata (looked for {STRATIFY_KEY_PRIORITY}); "
        f"pass stratify_key explicitly. Found keys: {sorted(keys)}"
    )


def _stratum_of(pair: BenchmarkQAPair, key: str) -> str:
    value = (pair.metadata or {}).get(key)
    if value is None:
        raise ValueError(f"question {pair.id!r} is missing stratify key {key!r} in metadata")
    return str(value)


def _hamilton_allocation(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment of ``total`` across strata by ``weights``.

    Guarantees the per-stratum allocations sum exactly to ``total`` while
    matching the weight proportions as closely as integer rounding allows.
    """
    pool = sum(weights.values())
    if pool == 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / pool for k, w in weights.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(alloc.values())
    order = sorted(weights, key=lambda k: (raw[k] - alloc[k], k), reverse=True)
    for k in order[:remainder]:
        alloc[k] += 1
    return alloc


def stratified_split(
    pairs: list[BenchmarkQAPair],
    *,
    stratify_key: str | None = None,
    holdout_size: int = DEFAULT_HOLDOUT_SIZE,
    seed: int,
) -> SplitResult:
    """Draw a stratified held-out slice; the rest usable becomes the reservoir.

    Rows with empty ``supporting_doc_ids`` are excluded first. Each stratum is
    apportioned its proportional share of the held-out slice (largest-remainder);
    a seeded shuffle draws that share, and every remaining usable row in the
    stratum goes to the disjoint optimization reservoir.
    """
    key = stratify_key or detect_stratify_key(pairs)

    usable = [p for p in pairs if p.supporting_doc_ids]
    n_excluded = len(pairs) - len(usable)
    if holdout_size > len(usable):
        raise ValueError(
            f"holdout_size ({holdout_size}) exceeds usable pool "
            f"({len(usable)} of {len(pairs)} after excluding {n_excluded} doc-less rows)"
        )

    by_stratum: dict[str, list[BenchmarkQAPair]] = {}
    for p in usable:
        by_stratum.setdefault(_stratum_of(p, key), []).append(p)
    counts = {s: len(v) for s, v in by_stratum.items()}

    holdout_alloc = _hamilton_allocation(holdout_size, counts)

    rng = random.Random(seed)
    holdout: list[BenchmarkQAPair] = []
    optimization: list[BenchmarkQAPair] = []
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum], key=lambda p: p.id)
        rng.shuffle(members)
        h = holdout_alloc[stratum]
        if h > len(members):
            raise ValueError(f"stratum {stratum!r} has {len(members)} usable rows but needs {h} holdout")
        holdout.extend(members[:h])
        optimization.extend(members[h:])

    holdout_ids = {p.id for p in holdout}
    opt_ids = {p.id for p in optimization}
    disjoint = holdout_ids.isdisjoint(opt_ids)
    if not disjoint:
        raise AssertionError("holdout and optimization slices overlap — split is broken")

    provenance = SplitProvenance(
        stratify_key=key,
        seed=seed,
        n_pool_total=len(pairs),
        n_excluded_no_docs=n_excluded,
        n_pool_usable=len(usable),
        holdout_size=len(holdout),
        opt_size=len(optimization),
        pool_distribution=dict(sorted(counts.items())),
        holdout_distribution=dict(sorted(Counter(_stratum_of(p, key) for p in holdout).items())),
        opt_distribution=dict(sorted(Counter(_stratum_of(p, key) for p in optimization).items())),
        disjoint=disjoint,
    )
    return SplitResult(holdout=holdout, optimization=optimization, provenance=provenance)


def write_splits(result: SplitResult, out_dir: Path) -> dict[str, Path]:
    """Write holdout_qa.json, optimization_qa.json, and split_provenance.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "holdout": out_dir / "holdout_qa.json",
        "optimization": out_dir / "optimization_qa.json",
        "provenance": out_dir / "split_provenance.json",
    }
    paths["holdout"].write_text(
        json.dumps([p.model_dump(mode="json") for p in result.holdout], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["optimization"].write_text(
        json.dumps([p.model_dump(mode="json") for p in result.optimization], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["provenance"].write_text(
        result.provenance.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return paths

"""The pinned HuggingFace revision must survive config -> runner -> adapter.

A rerun that silently prepares whatever is at the dataset's head builds a
different corpus from the published one, and nothing downstream would notice.
These tests pin the whole path: the YAML field is parsed, it reaches the
framework's ``prepare``, and it is recorded in ``bench_metadata.json`` so a
results tree self-documents which revision produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agentic_autorag_bench.benchmarks import runner as runner_mod
from agentic_autorag_bench.run import BenchConfig, _write_bench_metadata

REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


def _write_configs(tmp_path: Path, *, hf_revision: str | None) -> Path:
    (tmp_path / "project.yaml").write_text("meta:\n  corpus_path: ./corpus\n", encoding="utf-8")
    benchmark: dict = {
        "name": "hotpot_qa",
        "split": "validation",
        "sample_size": 2000,
        "prep_seed": 42,
        "output_dir": str(tmp_path / "data"),
    }
    if hf_revision is not None:
        benchmark["hf_revision"] = hf_revision
    config = {
        "project_config": "./project.yaml",
        "methods": ["random"],
        "seeds": [1],
        "budget": {"max_trials": 2},
        "benchmark": benchmark,
        "hold_out": {"limit": None, "judge_model": None, "concurrency": 8},
        "output_root": str(tmp_path / "out"),
    }
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_revision_parsed_from_config(tmp_path: Path) -> None:
    bench = BenchConfig.load(_write_configs(tmp_path, hf_revision=REVISION))
    assert bench.benchmark.hf_revision == REVISION


def test_revision_absent_is_none_not_an_error(tmp_path: Path) -> None:
    """Dev configs may omit the pin; that must stay valid and mean "head"."""
    bench = BenchConfig.load(_write_configs(tmp_path, hf_revision=None))
    assert bench.benchmark.hf_revision is None


def test_revision_forwarded_to_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(runner_mod, "prepare_benchmark", lambda **kw: seen.update(kw))

    runner_mod.BenchmarkRunner(
        name="hotpot_qa",
        output_dir=tmp_path / "data",
        split="validation",
        sample_size=2000,
        seed=42,
        hf_revision=REVISION,
    ).prepare()

    assert seen["hf_revision"] == REVISION
    assert seen["seed"] == 42
    assert seen["sample_size"] == 2000


def test_revision_recorded_in_bench_metadata(tmp_path: Path) -> None:
    bench = BenchConfig.load(_write_configs(tmp_path, hf_revision=REVISION))
    _write_bench_metadata(bench.output_root, bench)

    meta = json.loads((bench.output_root / "bench_metadata.json").read_text())
    assert meta["benchmark"]["hf_revision"] == REVISION

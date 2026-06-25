"""Tests for the ``agentic_opro`` naive-LLM-proposer baseline registration.

``agentic_opro`` reuses ``AgenticOptimizer`` with KB off, diagnosis off, and the
``opro`` (compact-history) flag on. These tests pin the registration + flag
wiring; the compact-history behavior itself is tested in the framework repo
(``Agentic-AutoRAG/tests/test_opro.py``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agentic_autorag_bench.methods.agentic import AgenticOptimizer
from agentic_autorag_bench.run import ALL_METHODS, STOCHASTIC_METHODS, BenchConfig, _build_optimizer


def _bench_stub() -> SimpleNamespace:
    return SimpleNamespace(project_config_path=Path("project.yaml"))


def test_agentic_opro_registered() -> None:
    assert "agentic_opro" in STOCHASTIC_METHODS
    assert "agentic_opro" in ALL_METHODS


def test_build_optimizer_wires_opro_flags(tmp_path: Path) -> None:
    """OPRO = AgenticOptimizer with opro on, KB off, diagnosis off, score mode."""
    opt = _build_optimizer(
        "agentic_opro",
        project=None,
        bench=_bench_stub(),
        output_dir=tmp_path / "agentic_opro" / "seed_1",
        resume=False,
    )
    assert isinstance(opt, AgenticOptimizer)
    assert opt.name == "agentic_opro"
    assert opt.opro is True
    assert opt.use_knowledge_base is False
    assert opt.use_diagnosis is False
    assert opt.cost_aware is False


def test_build_optimizer_score_keeps_kb_and_diagnosis(tmp_path: Path) -> None:
    """The headline method keeps KB + diagnosis and is not in OPRO mode."""
    opt = _build_optimizer(
        "agentic_score",
        project=None,
        bench=_bench_stub(),
        output_dir=tmp_path / "agentic_score" / "seed_1",
        resume=False,
    )
    assert opt.opro is False
    assert opt.use_knowledge_base is True
    assert opt.use_diagnosis is True


def test_build_optimizer_nodiag_keeps_kb_drops_diagnosis(tmp_path: Path) -> None:
    """nodiag isolates the diagnosis loop: KB on, diagnosis off, OPRO off."""
    opt = _build_optimizer(
        "agentic_nodiag",
        project=None,
        bench=_bench_stub(),
        output_dir=tmp_path / "agentic_nodiag" / "seed_1",
        resume=False,
    )
    assert opt.opro is False
    assert opt.use_knowledge_base is True
    assert opt.use_diagnosis is False


def _write_bench_config(tmp_path: Path, methods: list[str], checkpoints: dict | None = None) -> Path:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(yaml.safe_dump({"meta": {"output_dir": str(tmp_path / "_cache")}}))
    config = {
        "project_config": str(project_yaml),
        "methods": methods,
        "seeds": [1],
        "budget": {"max_trials": 40},
        "benchmark": {
            "name": "hotpot_qa",
            "split": "validation",
            "sample_size": 100,
            "prep_seed": 42,
            "output_dir": str(tmp_path / "_data"),
        },
        "hold_out": {"limit": 10, "judge_model": "test", "concurrency": 1},
        "output_root": str(tmp_path / "results"),
    }
    if checkpoints is not None:
        config["checkpoints"] = checkpoints
    config_yaml = tmp_path / "bench_config.yaml"
    config_yaml.write_text(yaml.safe_dump(config))
    return config_yaml


def test_benchconfig_load_accepts_opro(tmp_path: Path) -> None:
    path = _write_bench_config(
        tmp_path,
        methods=["agentic_score", "agentic_nodiag", "agentic_opro", "motpe", "motpe_warm", "random"],
        checkpoints={"agentic_score": [10, 20], "agentic_opro": [10, 20]},
    )
    bench = BenchConfig.load(path)
    assert "agentic_opro" in bench.methods
    assert bench.checkpoints["agentic_opro"] == [10, 20]


def test_benchconfig_load_rejects_unknown_method(tmp_path: Path) -> None:
    path = _write_bench_config(tmp_path, methods=["agentic_bogus", "random"])
    with pytest.raises(ValueError, match="Unknown methods"):
        BenchConfig.load(path)

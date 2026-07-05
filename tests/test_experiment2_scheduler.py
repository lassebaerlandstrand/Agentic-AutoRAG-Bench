"""Tests for the Experiment-2 (UniDoc Pareto) scheduler DAG + disk-truth oracle.

The scheduler is a standalone script under ``scripts/`` (not part of the package),
so it's loaded here by path. These tests exercise the pure, network-free pieces:
DAG construction (the ``motpe_warm -> random`` warm gate + FINALIZE fan-in) and
the completion predicates (SETUP marker / ``optimizer_meta.json`` / ``hypervolume.json``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_scheduler():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_experiment2.py"
    spec = importlib.util.spec_from_file_location("run_experiment2", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module's namespace for
    # string annotations (`from __future__ import annotations`).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _exp(sched, tmp_path, methods, seeds, max_trials=30):
    out = tmp_path / "out"
    return sched.Experiment(
        config_path=tmp_path / "c.yaml",
        project_config_path=tmp_path / "p.yaml",
        output_root=out,
        shared_cache_dir=out / ".shared_cache",
        max_trials=max_trials,
        methods=list(methods),
        seeds=list(seeds),
    )


def test_build_units_warm_gate_and_finalize_fanin(tmp_path) -> None:
    sched = _load_scheduler()
    exp = _exp(sched, tmp_path, ["agentic_cost", "random", "motpe_warm", "motpe"], [1])
    units = sched.build_units(exp)

    assert "setup" in units
    assert "finalize" in units
    # every method cell depends on SETUP (single-writer warmup)
    assert units["agentic_cost/seed_1"].deps == ["setup"]
    assert units["motpe/seed_1"].deps == ["setup"]
    # motpe_warm additionally depends on the same-seed random cell (transfer prior)
    assert units["motpe_warm/seed_1"].deps == ["setup", "random/seed_1"]
    # finalize waits on every method cell
    assert set(units["finalize"].deps) == {
        "agentic_cost/seed_1",
        "random/seed_1",
        "motpe_warm/seed_1",
        "motpe/seed_1",
    }


def test_build_units_multi_seed_warm_gate_is_per_seed(tmp_path) -> None:
    sched = _load_scheduler()
    exp = _exp(sched, tmp_path, ["random", "motpe_warm"], [1, 2])
    units = sched.build_units(exp)
    assert units["motpe_warm/seed_1"].deps == ["setup", "random/seed_1"]
    assert units["motpe_warm/seed_2"].deps == ["setup", "random/seed_2"]
    # 2 seeds x 2 methods + setup + finalize
    assert len(units) == 6


def test_test_subset_dag_matches_test_run(tmp_path) -> None:
    sched = _load_scheduler()
    exp = _exp(sched, tmp_path, ["agentic_cost", "motpe"], [1])
    units = sched.build_units(exp)
    assert set(units) == {"setup", "agentic_cost/seed_1", "motpe/seed_1", "finalize"}
    # no warm gate in this subset
    assert units["motpe/seed_1"].deps == ["setup"]


def test_completion_oracle_setup_method_finalize(tmp_path) -> None:
    sched = _load_scheduler()
    exp = _exp(sched, tmp_path, ["motpe"], [1])
    units = sched.build_units(exp)
    setup_u, method_u, fin_u = units["setup"], units["motpe/seed_1"], units["finalize"]

    # SETUP: marker gate
    assert sched.is_unit_complete(exp, setup_u) is False
    exp.shared_cache_dir.mkdir(parents=True)
    (exp.shared_cache_dir / sched.SETUP_MARKER).write_text("ok")
    assert sched.is_unit_complete(exp, setup_u) is True

    # METHOD: optimizer_meta.json n_trials_completed >= max_trials (NOT benchmark_results.json)
    md = exp.output_root / "motpe" / "seed_1"
    md.mkdir(parents=True)
    assert sched.is_unit_complete(exp, method_u) is False
    (md / "optimizer_meta.json").write_text(json.dumps({"n_trials_completed": 29}))
    assert sched.is_unit_complete(exp, method_u) is False
    (md / "optimizer_meta.json").write_text(json.dumps({"n_trials_completed": 30}))
    assert sched.is_unit_complete(exp, method_u) is True

    # FINALIZE: hypervolume.json gate
    assert sched.is_unit_complete(exp, fin_u) is False
    (exp.output_root / "hypervolume.json").write_text("{}")
    assert sched.is_unit_complete(exp, fin_u) is True


def test_build_argv_per_kind(tmp_path) -> None:
    sched = _load_scheduler()
    exp = _exp(sched, tmp_path, ["agentic_cost", "motpe"], [1])
    units = sched.build_units(exp)

    setup_argv = sched.build_argv(exp, units["setup"])
    assert "pareto" in setup_argv
    assert "--setup-only" in setup_argv

    method_argv = sched.build_argv(exp, units["agentic_cost/seed_1"])
    assert "--methods" in method_argv
    assert method_argv[method_argv.index("--methods") + 1] == "agentic_cost"
    assert "--seed" in method_argv
    assert method_argv[method_argv.index("--seed") + 1] == "1"
    assert "--resume" in method_argv

    fin_argv = sched.build_argv(exp, units["finalize"])
    assert "--figure-only" in fin_argv

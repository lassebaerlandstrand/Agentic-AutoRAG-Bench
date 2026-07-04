"""Smoke tests for the Pillar 1 validity harness (pure logic + fixtures).

The real correlation runs later against the full ``random`` trajectory; here we
only verify the trajectory loader, sampler, validity math, and project rewrite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pillar1_validity.py"
_spec = importlib.util.spec_from_file_location("pillar1_validity", _SCRIPT)
p1 = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve the module under __future__ annotations.
sys.modules["pillar1_validity"] = p1
_spec.loader.exec_module(p1)


def _write_history(results_dir: Path, seed: str, records: list[dict]) -> None:
    d = results_dir / seed / "details"
    d.mkdir(parents=True, exist_ok=True)
    (d / "history.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_load_trajectory_reads_all_seeds(tmp_path: Path) -> None:
    results = tmp_path / "random"
    _write_history(results, "seed_1", [{"trial_number": 1, "config": {"top_k": 5}, "answer_accuracy": 0.4}])
    _write_history(results, "seed_2", [{"trial_number": 1, "config": {"top_k": 9}, "answer_accuracy": 0.6}])
    points = p1.load_trajectory(results)
    assert len(points) == 2
    assert {p.validation_score for p in points} == {0.4, 0.6}


def test_load_trajectory_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        p1.load_trajectory(tmp_path / "nope")


def test_sample_configs_spans_range(tmp_path: Path) -> None:
    points = [
        p1.TrajectoryPoint(trial_number=i, seed="seed_1", config={"top_k": i}, validation_score=i / 100)
        for i in range(100)
    ]
    picked = p1.sample_configs(points, n=5)
    scores = [p.validation_score for p in picked]
    assert len(picked) == 5
    assert scores[0] == pytest.approx(0.0)  # lowest
    assert scores[-1] == pytest.approx(0.99)  # highest
    assert scores == sorted(scores)


def test_sample_configs_returns_all_when_small() -> None:
    points = [p1.TrajectoryPoint(1, "seed_1", {"top_k": 5}, 0.5)]
    assert len(p1.sample_configs(points, n=18)) == 1


def test_compute_validity_perfect_correlation() -> None:
    validation = [0.1, 0.3, 0.5, 0.7, 0.9]
    report = p1.compute_validity(validation, validation)
    assert report.spearman_rho == pytest.approx(1.0)
    assert report.kendall_tau == pytest.approx(1.0)
    assert report.selection_regret == pytest.approx(0.0)  # self-best == validation-best


def test_compute_validity_anti_correlation_has_regret() -> None:
    validation = [0.1, 0.3, 0.5, 0.7, 0.9]
    self_scores = [0.9, 0.7, 0.5, 0.3, 0.1]  # inverted
    report = p1.compute_validity(validation, self_scores)
    assert report.spearman_rho == pytest.approx(-1.0)
    # self-best (idx 0) has the WORST validation score → regret = 0.9 - 0.1.
    assert report.selection_regret == pytest.approx(0.8)


def test_compute_validity_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        p1.compute_validity([0.1, 0.2], [0.1])


def test_build_self_exam_project_strips_custom_exam_path(tmp_path: Path) -> None:
    paper = tmp_path / "hotpot_paper_project.yaml"
    paper.write_text(
        yaml.safe_dump(
            {
                "meta": {"project_name": "x", "output_dir": "./results/.shared_cache"},
                "examiner": {"exam_size": 100, "custom_exam_path": "./validation_exam.json"},
                "agent": {"optimizer_model": "m", "examiner_model": "m", "judge_model": "m"},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "selfexam_cache"
    written = p1.build_self_exam_project(paper, out_dir)
    raw = yaml.safe_load(written.read_text())
    assert "custom_exam_path" not in raw["examiner"]  # generate self-exam, not the validation exam
    assert raw["meta"]["output_dir"] == str(out_dir)
    assert raw["examiner"]["exam_size"] == 100  # other fields preserved

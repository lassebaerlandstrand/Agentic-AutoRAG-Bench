"""Pillar 1 — exam-generation validity harness.

Validates the self-generated exam as a quality signal by correlating its score
against the validation-exam score across configs drawn from ``random``'s
Pillar-2 trajectory. It does NOT re-run the optimizer or the held-out gold: it
reads each ``random`` trial's (config, validation-exam accuracy) from
``history.jsonl``, samples configs across the score range, re-scores each on the
self-exam via the framework Orchestrator, and reports Spearman rho / Kendall tau
+ selection regret.

The self-exam is generated into a separate ``meta.output_dir`` (a sibling of the
Pillar-2 shared cache) from a copy of the dataset's paper project YAML with
``examiner.custom_exam_path`` removed — so it can't collide with the validation
exam the headline optimizes against.

Pure functions (``load_trajectory`` / ``sample_configs`` / ``compute_validity`` /
``build_self_exam_project``) are import-safe and unit-tested; only
``score_configs_on_self_exam`` and ``main`` touch a live Orchestrator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from scipy.stats import kendalltau, spearmanr

DEFAULT_SAMPLE_SIZE = 18


@dataclass
class TrajectoryPoint:
    trial_number: int
    seed: str
    config: dict
    validation_score: float


def load_trajectory(random_results_dir: Path) -> list[TrajectoryPoint]:
    """Read every ``seed_*/details/history.jsonl`` under a ``random`` results dir.

    Each history record carries the trial ``config`` and its ``answer_accuracy``
    on the optimization exam — which, in the Pillar-2 setup, is the validation exam.
    """
    random_results_dir = Path(random_results_dir)
    points: list[TrajectoryPoint] = []
    histories = sorted(random_results_dir.glob("seed_*/details/history.jsonl"))
    if not histories:
        raise FileNotFoundError(f"no seed_*/details/history.jsonl under {random_results_dir}")
    for hist in histories:
        seed = hist.parent.parent.name
        for line in hist.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "config" not in rec or "answer_accuracy" not in rec:
                continue
            points.append(
                TrajectoryPoint(
                    trial_number=int(rec.get("trial_number", 0)),
                    seed=seed,
                    config=rec["config"],
                    validation_score=float(rec["answer_accuracy"]),
                )
            )
    return points


def sample_configs(points: list[TrajectoryPoint], n: int = DEFAULT_SAMPLE_SIZE) -> list[TrajectoryPoint]:
    """Pick ``n`` configs spread evenly across the validation score range.

    Deterministic: sorts by score and takes evenly-spaced ranks, so the sample
    spans low/mid/high without an RNG. Returns everything when the pool is
    already at or below ``n``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if len(points) <= n:
        return list(points)
    ordered = sorted(points, key=lambda p: (p.validation_score, p.seed, p.trial_number))
    picked: list[TrajectoryPoint] = []
    seen: set[int] = set()
    for i in range(n):
        idx = round(i * (len(ordered) - 1) / (n - 1))
        if idx not in seen:
            seen.add(idx)
            picked.append(ordered[idx])
    return picked


@dataclass
class ValidityReport:
    n: int
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    selection_regret: float
    validation_scores: list[float]
    self_scores: list[float]


def compute_validity(validation_scores: list[float], self_scores: list[float]) -> ValidityReport:
    """Rank-correlate self-exam vs validation-exam scores + selection regret.

    ``selection_regret`` is the validation accuracy given up by picking the
    self-exam-best config instead of the validation-best config, among the sample —
    the metric that survives even a weak rho.
    """
    if len(validation_scores) != len(self_scores):
        raise ValueError("validation_scores and self_scores must be the same length")
    if len(validation_scores) < 2:
        raise ValueError("need at least 2 points to correlate")

    rho, rho_p = spearmanr(validation_scores, self_scores)
    tau, tau_p = kendalltau(validation_scores, self_scores)
    best_validation = max(validation_scores)
    self_best_idx = max(range(len(self_scores)), key=lambda i: self_scores[i])
    regret = best_validation - validation_scores[self_best_idx]
    return ValidityReport(
        n=len(validation_scores),
        spearman_rho=float(rho),
        spearman_p=float(rho_p),
        kendall_tau=float(tau),
        kendall_p=float(tau_p),
        selection_regret=float(regret),
        validation_scores=list(validation_scores),
        self_scores=list(self_scores),
    )


def build_self_exam_project(paper_project_path: Path, self_exam_output_dir: Path) -> Path:
    """Write a self-exam variant of the paper project YAML.

    Strips ``examiner.custom_exam_path`` (so the Orchestrator generates the
    corpus self-exam) and repoints ``meta.output_dir`` at a dedicated dir so the
    self-exam cache never collides with the validation exam run.
    """
    paper_project_path = Path(paper_project_path)
    self_exam_output_dir = Path(self_exam_output_dir)
    raw = yaml.safe_load(paper_project_path.read_text(encoding="utf-8"))
    raw.setdefault("examiner", {}).pop("custom_exam_path", None)
    raw.setdefault("meta", {})["output_dir"] = str(self_exam_output_dir)
    self_exam_output_dir.mkdir(parents=True, exist_ok=True)
    out = self_exam_output_dir / (paper_project_path.stem + "_selfexam.yaml")
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out


async def score_configs_on_self_exam(self_exam_project_path: Path, configs: list[dict]) -> list[float]:
    """Re-score each config on the self-exam via a fresh Orchestrator (real work)."""
    from agentic_autorag.config.loader import load_config
    from agentic_autorag.config.models import TrialConfig
    from agentic_autorag.litellm_runtime import configure_litellm_runtime, install_model_aliases
    from agentic_autorag.orchestrator import Orchestrator

    project = load_config(str(self_exam_project_path))
    configure_litellm_runtime(project.model_aliases)
    install_model_aliases(project.model_aliases)

    orch = Orchestrator(str(self_exam_project_path))
    orch.evaluator.quiet_per_question = True
    scores: list[float] = []
    try:
        await orch.setup()  # generates the self-exam once
        for cfg in configs:
            result = await orch.evaluate_trial(TrialConfig(**cfg))
            scores.append(result.answer_accuracy)
    finally:
        await orch.cleanup()
    return scores


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pillar 1 exam-validity harness")
    p.add_argument("--random-results", required=True, type=Path, help="results_*/random dir with seed_*/details")
    p.add_argument("--paper-project", required=True, type=Path, help="dataset's *_paper_project.yaml")
    p.add_argument("--self-exam-output", required=True, type=Path, help="dedicated output dir for the self-exam run")
    p.add_argument("--output", required=True, type=Path, help="destination JSON for the validity report")
    p.add_argument("-n", "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    points = load_trajectory(args.random_results)
    sampled = sample_configs(points, n=args.sample_size)
    self_project = build_self_exam_project(args.paper_project, args.self_exam_output)
    self_scores = asyncio.run(score_configs_on_self_exam(self_project, [p.config for p in sampled]))
    report = compute_validity([p.validation_score for p in sampled], self_scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(f"Pillar 1 validity ({report.n} configs) -> {args.output}")
    print(f"  Spearman rho = {report.spearman_rho:.3f} (p={report.spearman_p:.3g})")
    print(f"  Kendall  tau = {report.kendall_tau:.3f} (p={report.kendall_p:.3g})")
    print(f"  selection regret = {report.selection_regret:.3f}")


if __name__ == "__main__":
    main()

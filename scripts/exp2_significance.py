"""Exp-2 (Pareto) per-seed significance tests behind the paper's Section 5.2 paragraph.

Recomputes, from ``experiment-2/unidoc/{method}/seed_*/details/history.jsonl``,
every number in the run-by-run significance paragraph:

1. **Peak accuracy** — each seed's best exam accuracy over its 30 trials,
   agent vs each baseline (two-sided Mann-Whitney U, Holm-corrected).
2. **Cost to reach the target** — each seed's cheapest trial reaching the
   strongest baseline's median peak accuracy, with seeds that never reach it
   ranked as the most expensive (censored), agent vs each baseline.
3. **Per-budget attainment** — the same test repeated on every budget of the
   240-point log grid that ``plots.py`` uses for the attainment figure,
   reporting the budget above which the agent's lead is significant against
   all baselines at every grid point, and the budget below which no
   comparison is significant in either direction.

Trial filtering matches ``plots._load_trial_points``: rows with missing
accuracy or non-positive cost are dropped. Writes a markdown summary next to
``hypervolume.json`` and prints a paste block for the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

GRID_POINTS = 240  # keep in sync with plots._ATTAINMENT_GRID_POINTS
ALPHA = 0.05


def load_seed_points(method_dir: Path) -> list[np.ndarray]:
    """Per seed: (n_trials, 2) array of (cost_per_query, answer_accuracy)."""
    seeds = sorted(method_dir.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    out: list[np.ndarray] = []
    for seed_dir in seeds:
        rows = []
        with open(seed_dir / "details" / "history.jsonl") as fh:
            for line in fh:
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated final line, as plots.py does
                cost = t.get("mean_llm_cost_per_query_usd")
                acc = t.get("answer_accuracy")
                if cost is None or cost <= 0 or acc is None:
                    continue
                rows.append((float(cost), float(acc)))
        if rows:
            out.append(np.array(rows))
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def prob_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """P(random a-seed beats random b-seed), ties counted half."""
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def mwu_rows(a: np.ndarray, baselines: dict[str, np.ndarray], *, larger_is_better: bool) -> list[dict]:
    """Two-sided MWU of ``a`` against each baseline sample, Holm across the family."""
    rows = []
    for name, b in baselines.items():
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        better = prob_improvement(a, b) if larger_is_better else prob_improvement(-a, -b)
        rows.append({"baseline": name, "p": p, "p_improvement": better,
                     "median_baseline": float(np.median(b))})
    for row, adj in zip(rows, holm([r["p"] for r in rows]), strict=True):
        row["p_holm"] = adj
    return rows


def peak_per_seed(seed_points: list[np.ndarray]) -> np.ndarray:
    return np.array([pts[:, 1].max() for pts in seed_points])


def cost_to_reach(seed_points: list[np.ndarray], target: float) -> np.ndarray:
    """Per seed, cheapest trial with accuracy >= target; +inf when never reached."""
    out = []
    for pts in seed_points:
        hit = pts[pts[:, 1] >= target]
        out.append(float(hit[:, 0].min()) if len(hit) else np.inf)
    return np.array(out)


def attainment(seed_points: list[np.ndarray], budget: float) -> np.ndarray:
    """Per seed, best accuracy among trials costing <= budget (0.0 if none)."""
    vals = []
    for pts in seed_points:
        ok = pts[pts[:, 0] <= budget]
        vals.append(float(ok[:, 1].max()) if len(ok) else 0.0)
    return np.array(vals)


def budget_boundaries(data: dict[str, list[np.ndarray]], agent: str) -> dict:
    """Significance structure of agent-vs-baseline attainment over the cost grid."""
    all_costs = np.concatenate([pts[:, 0] for m in data for pts in data[m]])
    grid = np.logspace(np.log10(all_costs.min()), np.log10(all_costs.max()), GRID_POINTS)
    sig_lead = np.ones(len(grid), dtype=bool)
    any_sig = np.zeros(len(grid), dtype=bool)
    for i, budget in enumerate(grid):
        a = attainment(data[agent], budget)
        for name in data:
            if name == agent:
                continue
            b = attainment(data[name], budget)
            if np.ptp(np.concatenate([a, b])) == 0:
                p, lead = 1.0, 0.5
            else:
                p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                lead = prob_improvement(a, b)
            if p >= ALPHA or lead <= 0.5:
                sig_lead[i] = False
            if p < ALPHA:
                any_sig[i] = True
    # smallest budget from which the lead is significant at every larger grid point
    tail_ok = np.logical_and.accumulate(sig_lead[::-1])[::-1]
    lead_from = float(grid[np.argmax(tail_ok)]) if tail_ok.any() else None
    # largest budget below which no comparison is significant in either direction
    head_ok = np.logical_and.accumulate(~any_sig)
    none_below = float(grid[np.max(np.nonzero(head_ok))]) if head_ok.any() else None
    return {"grid_points": len(grid), "grid_min": float(grid[0]), "grid_max": float(grid[-1]),
            "sig_lead_from_budget": lead_from, "no_significance_below_budget": none_below}


def fmt_rows(rows: list[dict]) -> str:
    lines = ["| baseline | median | MWU p | Holm p | P(agent better) |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['baseline']} | {r['median_baseline']:.4g} | {r['p']:.4f} "
                     f"| {r['p_holm']:.4f} | {r['p_improvement']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=Path("experiment-2/unidoc"))
    parser.add_argument("--agent", default="agentic_cost")
    parser.add_argument("--baselines", nargs="+", default=["motpe_warm", "motpe", "random"])
    parser.add_argument("--strongest-baseline", default="motpe_warm",
                        help="method whose median per-seed peak defines the cost-to-reach target")
    args = parser.parse_args()

    data = {m: load_seed_points(args.output_root / m) for m in [args.agent, *args.baselines]}
    for m, pts in data.items():
        if not pts:
            raise SystemExit(f"no seed data under {args.output_root / m}")

    agent_peaks = peak_per_seed(data[args.agent])
    target = float(np.median(peak_per_seed(data[args.strongest_baseline])))

    peak_rows = mwu_rows(agent_peaks, {m: peak_per_seed(data[m]) for m in args.baselines},
                         larger_is_better=True)
    agent_cost = cost_to_reach(data[args.agent], target)
    cost_rows = mwu_rows(agent_cost, {m: cost_to_reach(data[m], target) for m in args.baselines},
                         larger_is_better=False)
    reach = {m: int(np.isfinite(cost_to_reach(data[m], target)).sum()) for m in data}
    bounds = budget_boundaries(data, args.agent)

    report = [
        "# Exp-2 per-seed significance (regenerated by scripts/exp2_significance.py)",
        "",
        f"Agent `{args.agent}`, {len(data[args.agent])} seeds per method. "
        f"Two-sided Mann-Whitney U, Holm-corrected per family of {len(args.baselines)}.",
        "",
        "## Peak exam accuracy per seed",
        f"Agent peaks: min {agent_peaks.min():.3f}, median {np.median(agent_peaks):.3f}, "
        f"max {agent_peaks.max():.3f}.",
        "",
        fmt_rows(peak_rows),
        "",
        f"## Cost to reach {target:.1%} (strongest baseline's median peak)",
        "Seeds reaching it: " + ", ".join(f"{m} {reach[m]}/{len(data[m])}" for m in data) + ".",
        f"Agent: median \\${np.median(agent_cost):.5f}, max \\${agent_cost.max():.5f} per query. "
        "Seeds that never reach it are ranked most expensive (censored).",
        "",
        fmt_rows(cost_rows),
        "",
        "## Per-budget attainment on the shared "
        f"{bounds['grid_points']}-point log cost grid "
        f"[\\${bounds['grid_min']:.6f}, \\${bounds['grid_max']:.6f}]",
        f"Agent lead significant vs every baseline at every grid budget from "
        f"\\${bounds['sig_lead_from_budget']:.6f} upward.",
        f"No agent-vs-baseline comparison significant in either direction at any grid budget "
        f"up to \\${bounds['no_significance_below_budget']:.6f}.",
        "",
    ]
    text = "\n".join(report)
    out_path = args.output_root / "significance.md"
    out_path.write_text(text)
    print(text)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()

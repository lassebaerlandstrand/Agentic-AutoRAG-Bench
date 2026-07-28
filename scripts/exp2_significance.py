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
4. **Median-curve milestones** — the descriptive facts about the median
   attainment curve the figure draws: where each method's median curve peaks,
   what the agent pays to match the strongest baseline's peak, the cost from
   which the agent's median curve leads every baseline, and the band below it
   where the strongest baseline leads instead. These back the prose sentences
   that precede the significance paragraph, which were previously read off the
   figure by hand.

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


def median_curves(data: dict[str, list[np.ndarray]], costs: np.ndarray) -> dict[str, np.ndarray]:
    """Point-wise median across seeds of each method's attainment curve."""
    return {m: np.array([float(np.median(attainment(data[m], c))) for c in costs]) for m in data}


def first_cost_at(costs: np.ndarray, curve: np.ndarray, level: float) -> float | None:
    """Cheapest cost at which ``curve`` has reached ``level``."""
    idx = np.nonzero(curve >= level - 1e-12)[0]
    return float(costs[idx[0]]) if len(idx) else None


def median_milestones(data: dict[str, list[np.ndarray]], agent: str, strongest: str) -> dict:
    """Descriptive facts about the median attainment curves the figure draws.

    Evaluated at the trial costs themselves, not on the 240-point plotting grid.
    A median attainment curve is a step function that can only change at a trial
    cost, so the grid pins each milestone only to within one step, which is
    enough to shift a rounded figure like \\$0.00023 vs \\$0.00024. The grid stays
    the right resolution for ``budget_boundaries``, whose per-budget tests are
    defined on the grid the figure is drawn on.
    """
    costs = np.unique(np.concatenate([pts[:, 0] for m in data for pts in data[m]]))
    curves = median_curves(data, costs)
    peaks = {m: float(curves[m].max()) for m in curves}
    at_peak = {m: first_cost_at(costs, curves[m], peaks[m]) for m in curves}
    target = peaks[strongest]

    # cheapest cost from which the agent's median curve is above every baseline
    # and never falls back behind
    lead = np.ones(len(costs), dtype=bool)
    for m in curves:
        if m != agent:
            lead &= curves[agent] > curves[m]
    tail = np.logical_and.accumulate(lead[::-1])[::-1]
    lead_from = float(costs[int(np.argmax(tail))]) if tail.any() else None

    # the band below that, where the strongest baseline's median curve is ahead
    ahead = np.nonzero(curves[strongest] > curves[agent])[0]
    band = None
    if len(ahead):
        band = {
            "lo": float(costs[ahead[0]]),
            "hi": float(costs[ahead[-1]]),
            "agent_lo": float(curves[agent][ahead].min()),
            "agent_hi": float(curves[agent][ahead].max()),
            "strongest_lo": float(curves[strongest][ahead].min()),
            "strongest_hi": float(curves[strongest][ahead].max()),
        }
    return {"peaks": peaks, "at_peak": at_peak, "target": target,
            "match_cost": first_cost_at(costs, curves[agent], target),
            "lead_from": lead_from, "band": band}


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
    mil = median_milestones(data, args.agent, args.strongest_baseline)

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
        "## Median attainment curves (the figure's line), evaluated at the trial costs",
        "A median attainment curve steps only at a trial cost, so these are exact rather "
        "than pinned to the nearest plotting-grid point.",
        "",
        "| method | median-curve peak | cheapest cost reaching it |",
        "|---|---|---|",
        *[f"| {m} | {mil['peaks'][m]:.1%} | \\${mil['at_peak'][m]:.6f} |" for m in data],
        "",
        f"Agent matches `{args.strongest_baseline}`'s median peak of {mil['target']:.1%} at "
        f"\\${mil['match_cost']:.6f}, "
        f"{mil['at_peak'][args.strongest_baseline] / mil['match_cost']:.2f}x cheaper than that "
        f"baseline pays for it. The agent's own median peak costs "
        f"{mil['at_peak'][args.strongest_baseline] / mil['at_peak'][args.agent]:.2f}x less than "
        "the baseline's peak.",
        f"Agent's median curve is above every baseline's from \\${mil['lead_from']:.6f} upward "
        "and never falls back behind.",
        *([f"Below that, `{args.strongest_baseline}` leads from \\${mil['band']['lo']:.6f} up to "
           "that crossover, where the agent's median runs "
           f"{mil['band']['agent_lo']:.0%}-{mil['band']['agent_hi']:.0%} and the baseline's "
           f"{mil['band']['strongest_lo']:.0%}-{mil['band']['strongest_hi']:.0%}."]
          if mil["band"] else []),
        "",
    ]
    text = "\n".join(report)
    out_path = args.output_root / "significance.md"
    out_path.write_text(text)
    print(text)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()

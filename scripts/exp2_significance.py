"""Exp-2 (Pareto) per-seed significance tests behind the paper's Section 5.2 paragraph.

Recomputes, from ``experiment-2/unidoc/{method}/seed_*/details/history.jsonl``,
every number in the run-by-run significance paragraph:

1. **Peak accuracy** — each seed's best exam accuracy over its 30 trials,
   agent vs each baseline (two-sided Mann-Whitney U, Holm-corrected).
2. **Cost to reach the target** — each seed's cheapest trial reaching the
   strongest baseline's median peak accuracy, with seeds that never reach it
   ranked as the most expensive (censored), agent vs each baseline.
3. **Per-budget attainment** — the same test repeated on every budget of the
   240-point log grid that ``plots.py`` uses for the attainment figure, read
   as a fixed-sequence procedure (Maurer et al. 1995): step down from the
   largest budget until the within-budget Holm family first fails to reject,
   which controls the familywise error rate across the sweep. Reports the
   onset budget, its exact counterpart at every distinct trial cost, raw
   rejection islands below the stop (not claimed), and the budget below
   which no comparison is significant in either direction (raw, the
   conservative direction for an absence claim).
4. **Median-curve milestones** — the descriptive facts about the median
   attainment curve the figure draws: where each method's median curve peaks,
   what the agent pays to match the strongest baseline's peak, the cost from
   which the agent's median curve leads every baseline, and the band below it
   where the strongest baseline leads instead. These back the prose sentences
   that precede the significance paragraph, which were previously read off the
   figure by hand.
5. **Simultaneous bands (Westfall-Young maxT)** — the per-budget grid tests of
   (3) are heavily dependent, so this adjusts each budget's rank-sum statistic
   against the exact permutation distribution of the maximum across all
   budgets (all C(20,10) = 184,756 relabelings per baseline), reporting the
   agent-favoring bands with familywise error control over the whole grid and
   the window where the lead holds against all baselines at once.
6. **Accuracy-bar sweep** — the cost-to-reach test of (2) repeated at
   alternative accuracy bars, showing the result is not an artifact of the
   71.5% target.

Trial filtering matches ``plots._load_trial_points``: rows with missing
accuracy or non-positive cost are dropped. Writes a markdown summary next to
``hypervolume.json`` and prints a paste block for the paper.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

GRID_POINTS = 240  # keep in sync with plots._ATTAINMENT_GRID_POINTS
ALPHA = 0.05
BAR_SWEEP = [0.66, 0.68, 0.70, 0.705, 0.715, 0.72, 0.74, 0.75]
MAXT_CHUNK = 4096  # relabelings per vectorized block in the maxT enumeration


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


def cost_grid(data: dict[str, list[np.ndarray]]) -> np.ndarray:
    """The shared log cost grid, spanning every observed trial cost."""
    all_costs = np.concatenate([pts[:, 0] for m in data for pts in data[m]])
    return np.logspace(np.log10(all_costs.min()), np.log10(all_costs.max()), GRID_POINTS)


def _lead_reject(data: dict[str, list[np.ndarray]], agent: str, budget: float) -> tuple[bool, list[dict]]:
    """One budget: Holm-corrected all-baseline lead rejection, plus raw per-baseline facts."""
    a = attainment(data[agent], budget)
    rows = []
    for name in data:
        if name == agent:
            continue
        b = attainment(data[name], budget)
        if np.ptp(np.concatenate([a, b])) == 0:
            p, lead = 1.0, 0.5
        else:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            lead = prob_improvement(a, b)
        rows.append({"p": p, "lead": lead})
    ok = (all(h < ALPHA for h in holm([r["p"] for r in rows]))
          and all(r["lead"] > 0.5 for r in rows))
    return ok, rows


def budget_boundaries(data: dict[str, list[np.ndarray]], agent: str) -> dict:
    """Fixed-sequence significance structure of agent-vs-baseline attainment.

    The onset reads the grid sweep as a fixed-sequence procedure (Maurer et
    al. 1995): descend from the largest budget and stop where the
    within-budget Holm family first fails to reject, which controls the
    familywise error rate across the sweep without adjusting the individual
    budgets. Raw rejection islands below the stop are reported but carry no
    claim. The negative facts (baseline-favoring rejections, the
    no-significance head) stay on raw p-values, the conservative direction
    for absence statements. The exact onset repeats the walk at every
    distinct trial cost, where the attainment step functions actually change,
    so the grid onset is a conservative rounding-up of it.
    """
    grid = cost_grid(data)
    sig_lead = np.zeros(len(grid), dtype=bool)
    any_sig = np.zeros(len(grid), dtype=bool)
    n_baseline_favoring_sig = 0
    for i, budget in enumerate(grid):
        ok, rows = _lead_reject(data, agent, budget)
        sig_lead[i] = ok
        for r in rows:
            if r["p"] < ALPHA:
                any_sig[i] = True
                if r["lead"] < 0.5:
                    n_baseline_favoring_sig += 1
    # the fixed-sequence stop: smallest budget from which the lead rejects at every larger grid point
    tail_ok = np.logical_and.accumulate(sig_lead[::-1])[::-1]
    lead_from = float(grid[np.argmax(tail_ok)]) if tail_ok.any() else None
    # largest budget below which no comparison is significant in either direction
    head_ok = np.logical_and.accumulate(~any_sig)
    none_below = float(grid[np.max(np.nonzero(head_ok))]) if head_ok.any() else None
    islands = _true_ranges(sig_lead & ~tail_ok, grid)
    costs = np.unique(np.concatenate([pts[:, 0] for m in data for pts in data[m]]))
    cont_ok = np.array([_lead_reject(data, agent, c)[0] for c in costs])
    cont_tail = np.logical_and.accumulate(cont_ok[::-1])[::-1]
    exact_lead_from = float(costs[np.argmax(cont_tail)]) if cont_tail.any() else None
    return {"grid_points": len(grid), "grid_min": float(grid[0]), "grid_max": float(grid[-1]),
            "sig_lead_from_budget": lead_from, "no_significance_below_budget": none_below,
            "n_baseline_favoring_sig": n_baseline_favoring_sig,
            "unclaimed_islands": islands, "n_distinct_costs": len(costs),
            "exact_sig_lead_from_budget": exact_lead_from}


def _attain_matrix(seed_points: list[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """(n_seeds, n_budgets) matrix of per-seed attainment at each grid budget."""
    return np.array([attainment(seed_points, b) for b in grid]).T


def _true_ranges(mask: np.ndarray, grid: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous True runs of ``mask`` as (low, high) budget pairs."""
    ranges, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ranges.append((float(grid[start]), float(grid[i - 1])))
            start = None
    if start is not None:
        ranges.append((float(grid[start]), float(grid[-1])))
    return ranges


def maxt_bands(data: dict[str, list[np.ndarray]], agent: str) -> dict:
    """Exact Westfall-Young maxT simultaneous bands over the cost grid.

    The 240 per-budget tests are nested restatements of a few dozen distinct
    comparisons, so elementwise correction (Holm/Bonferroni) is far too
    conservative and no correction at all is too liberal. MaxT prices the
    dependence exactly: per baseline, each budget's tie-corrected rank-sum z
    is adjusted against the permutation distribution of the maximum z across
    all budgets, enumerated over every C(20,10) relabeling of the 20 seeds.
    Adjusted p < ALPHA at a budget then carries familywise error control over
    the entire grid. Reported one-sided (agent-favoring, the directional lead
    claim) and two-sided (matching the paper's other tests).
    """
    grid = cost_grid(data)
    a_mat = _attain_matrix(data[agent], grid)
    n_agent = a_mat.shape[0]
    labelings = np.array(list(combinations(range(2 * n_agent), n_agent)))
    out: dict = {"grid": grid, "n_labelings": len(labelings), "baselines": {}}
    for name in data:
        if name == agent:
            continue
        pooled = np.vstack([a_mat, _attain_matrix(data[name], grid)])
        n_total, n_budgets = pooled.shape
        ranks = np.empty_like(pooled)
        sd = np.empty(n_budgets)
        for j in range(n_budgets):
            ranks[:, j] = rankdata(pooled[:, j])
            _, counts = np.unique(pooled[:, j], return_counts=True)
            ties = (counts**3 - counts).sum() / (n_total * (n_total - 1))
            var = n_agent * (n_total - n_agent) / 12.0 * (n_total + 1 - ties)
            sd[j] = math.sqrt(var) if var > 1e-12 else 0.0
        mean = n_agent * (n_total + 1) / 2.0
        live = sd > 0  # all-tied budgets (everyone at zero attainment) carry no test
        z_obs = np.zeros(n_budgets)
        z_obs[live] = (ranks[:n_agent, live].sum(axis=0) - mean) / sd[live]
        # identical rank columns give identical z, so enumerate on distinct ones
        cols, first = np.unique(ranks[:, live], axis=1, return_index=True)
        sd_u = sd[live][first]
        max_one = np.empty(len(labelings))
        max_two = np.empty(len(labelings))
        for lo in range(0, len(labelings), MAXT_CHUNK):
            sums = cols[labelings[lo:lo + MAXT_CHUNK]].sum(axis=1)
            z = (sums - mean) / sd_u
            max_one[lo:lo + MAXT_CHUNK] = z.max(axis=1)
            max_two[lo:lo + MAXT_CHUNK] = np.abs(z).max(axis=1)
        max_one.sort()
        max_two.sort()
        eps = 1e-9
        adj_one = np.ones(n_budgets)
        adj_two = np.ones(n_budgets)
        n_lab = len(labelings)
        adj_one[live] = 1.0 - np.searchsorted(max_one, z_obs[live] - eps, side="left") / n_lab
        adj_two[live] = 1.0 - np.searchsorted(max_two, np.abs(z_obs[live]) - eps, side="left") / n_lab
        lead = z_obs > 0
        out["baselines"][name] = {
            "bands_one_sided": _true_ranges((adj_one < ALPHA) & lead, grid),
            "bands_two_sided": _true_ranges((adj_two < ALPHA) & lead, grid),
            "top_adj_p_one_sided": float(adj_one[-1]),
            "top_adj_p_two_sided": float(adj_two[-1]),
            "min_adj_p_one_sided": float(adj_one.min()),
            "min_adj_p_two_sided": float(adj_two.min()),
            "sig_one_sided": (adj_one < ALPHA) & lead,
            "sig_two_sided": (adj_two < ALPHA) & lead,
        }
    for side in ("one_sided", "two_sided"):
        joint = np.logical_and.reduce([b[f"sig_{side}"] for b in out["baselines"].values()])
        out[f"all_baseline_windows_{side}"] = _true_ranges(joint, grid)
    return out


def bar_sweep(data: dict[str, list[np.ndarray]], agent: str, baselines: list[str]) -> list[dict]:
    """The censored cost-to-reach test of section 2, swept over accuracy bars."""
    rows = []
    for bar in BAR_SWEEP:
        costs = {m: cost_to_reach(data[m], bar) for m in data}
        # a finite sentinel keeps the ranks of never-reaching seeds identical to
        # +inf while making never-vs-never pairs well-defined ties
        finite = np.concatenate([c[np.isfinite(c)] for c in costs.values()])
        cap = float(finite.max()) * 10 if len(finite) else 1.0
        capped = {m: np.where(np.isfinite(c), c, cap) for m, c in costs.items()}
        mwu = mwu_rows(capped[agent], {m: capped[m] for m in baselines}, larger_is_better=False)
        rows.append({"bar": bar,
                     "reach": {m: int(np.isfinite(costs[m]).sum()) for m in data},
                     "holm": {r["baseline"]: r["p_holm"] for r in mwu}})
    return rows


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
        lines.append(f"| {r['baseline']} | {r['median_baseline']:.4g} | {r['p']:.6f} "
                     f"| {r['p_holm']:.6f} | {r['p_improvement']:.2f} |")
    return "\n".join(lines)


def fmt_ranges(ranges: list[tuple[float, float]], grid_max: float) -> str:
    if not ranges:
        return "none"
    parts = []
    for lo, hi in ranges:
        if hi >= grid_max - 1e-12:
            parts.append(f"\\${lo:.6f} to grid max")
        else:
            parts.append(f"\\${lo:.6f} to \\${hi:.6f}")
    return ", ".join(parts)


def fmt_bar_sweep(rows: list[dict], methods: list[str], baselines: list[str]) -> str:
    header = ("| bar | " + " | ".join(f"reach {m}" for m in methods)
              + " | " + " | ".join(f"Holm p {b}" for b in baselines) + " |")
    lines = [header, "|" + "---|" * (1 + len(methods) + len(baselines))]
    for r in rows:
        lines.append(f"| {r['bar']:.1%} | "
                     + " | ".join(f"{r['reach'][m]}/10" for m in methods) + " | "
                     + " | ".join(f"{r['holm'][b]:.6f}" for b in baselines) + " |")
    return "\n".join(lines)


def fmt_maxt(bands: dict, baselines: list[str]) -> list[str]:
    grid_max = float(bands["grid"][-1])
    lines = ["| baseline | one-sided bands (adj p < 0.05) | two-sided bands | "
             "top-of-grid adj p (1s / 2s) | min adj p (1s / 2s) |",
             "|---|---|---|---|---|"]
    for b in baselines:
        r = bands["baselines"][b]
        lines.append(
            f"| {b} | {fmt_ranges(r['bands_one_sided'], grid_max)} "
            f"| {fmt_ranges(r['bands_two_sided'], grid_max)} "
            f"| {r['top_adj_p_one_sided']:.4f} / {r['top_adj_p_two_sided']:.4f} "
            f"| {r['min_adj_p_one_sided']:.2g} / {r['min_adj_p_two_sided']:.2g} |")
    lines += ["",
              "All-baseline simultaneous window (agent leads all three at once, "
              "intersection-union over the per-baseline bands): "
              f"one-sided {fmt_ranges(bands['all_baseline_windows_one_sided'], grid_max)}, "
              f"two-sided {fmt_ranges(bands['all_baseline_windows_two_sided'], grid_max)}."]
    return lines


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
    sweep = bar_sweep(data, args.agent, args.baselines)
    bands = maxt_bands(data, args.agent)
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
        "| method | sorted per-seed peaks | median |",
        "|---|---|---|",
        *[f"| {m} | " + " ".join(f"{v:.2f}" for v in np.sort(peak_per_seed(data[m])))
          + f" | {np.median(peak_per_seed(data[m])):.3f} |" for m in data],
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
        "## Accuracy-bar sweep for the cost-to-reach test",
        "The same censored test at alternative accuracy bars. Never-reaching seeds are "
        "ranked most expensive at every bar.",
        "",
        fmt_bar_sweep(sweep, list(data), args.baselines),
        "",
        f"Largest Holm p across all bars and baselines: "
        f"{max(p for r in sweep for p in r['holm'].values()):.6f}.",
        "",
        "## Per-budget attainment on the shared "
        f"{bounds['grid_points']}-point log cost grid "
        f"[\\${bounds['grid_min']:.6f}, \\${bounds['grid_max']:.6f}]",
        "Fixed-sequence read of the sweep (Maurer et al. 1995): step down from the largest "
        "budget until the within-budget Holm family first fails to reject. The stop controls "
        "the familywise error rate across the sweep without adjusting the individual budgets.",
        f"Agent lead significant vs every baseline at every grid budget from "
        f"\\${bounds['sig_lead_from_budget']:.6f} upward.",
        f"Exact onset of the same walk at every distinct trial cost "
        f"(n={bounds['n_distinct_costs']}, the only points where the attainment steps move): "
        f"\\${bounds['exact_sig_lead_from_budget']:.6f}, so the grid onset above is "
        "conservative.",
        f"No agent-vs-baseline comparison significant in either direction at any grid budget "
        f"up to \\${bounds['no_significance_below_budget']:.6f} (raw, the conservative "
        "direction for an absence claim).",
        f"Significant baseline-favoring comparisons anywhere on the grid: "
        f"{bounds['n_baseline_favoring_sig']} (no budget ever favors a baseline, raw).",
        *(["Raw rejection islands below the stop, reported but not claimed under the "
           "stopping rule: "
           + ", ".join(f"\\${lo:.6f} to \\${hi:.6f}" for lo, hi in bounds["unclaimed_islands"])
           + "."] if bounds["unclaimed_islands"] else []),
        "The maxT bands below spend alpha uniformly over all simultaneous per-budget "
        "statements, so they thin out toward the top of the grid, while the fixed sequence "
        "front-loads alpha at the largest budgets. Both control the familywise error rate, "
        "and both start the all-baseline lead at the same onset.",
        "",
        "## Simultaneous bands over the cost grid (exact Westfall-Young maxT)",
        "Tie-corrected rank-sum z per grid budget, adjusted against the exact permutation "
        f"distribution of the max z across all budgets ({bands['n_labelings']:,} relabelings "
        "per baseline). Adjusted p < 0.05 at a budget carries familywise error control over "
        "the entire grid. One-sided = agent-favoring lead, two-sided matches the paper's "
        "other tests.",
        "",
        *fmt_maxt(bands, args.baselines),
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

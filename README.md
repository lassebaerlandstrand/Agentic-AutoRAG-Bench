# Agentic AutoRAG Benchmark Suite

This is the benchmark and reproduction code for the Agentic AutoRAG paper. It
compares our reasoning-agent optimizer against random search and MO-TPE (the
optimizer from syftr) on two experiments:

- **Experiment 1 (accuracy):** each optimizer tunes a RAG pipeline on HotpotQA,
  MuSiQue, and MultiHop-RAG, scored on a held-out slice of each dataset.
- **Experiment 2 (cost and quality):** the optimizers tune for accuracy and
  per-query cost together on UniDoc-Bench healthcare, producing a cost-quality
  Pareto frontier.

## Methods

| Key | Description |
|-----|-------------|
| `agentic_score` | Our reasoning-loop optimizer, tuning for accuracy. |
| `agentic_cost` | Our optimizer, tuning for accuracy and cost together. Used in Experiment 2. |
| `agentic_nokb_nodiag` | Our optimizer stripped to score-only, with no knowledge base and no diagnosis. A rival baseline in Experiment 1. |
| `agentic_nokb`, `agentic_nodiag` | Single-component ablations, with the knowledge base off or the diagnosis off. Hotpot only, run after the headline. |
| `random` | Random search. Also seeds the warm-started MO-TPE. |
| `motpe` | Optuna multivariate MO-TPE, the optimizer used by syftr. |
| `motpe_warm` | The same MO-TPE, warm-started from this run's random trials. |
| `qlognehvi` | A GP-BO reference (Barker et al.). Cited in the paper but not run. Needs `uv add ax-platform`. |

All methods search the same pipeline space and are scored the same way, so a
results table compares them directly. The agentic methods share one optimizer
class and differ only by config flags. The two MO-TPE methods share another
class, with a flag selecting accuracy-only or accuracy-and-cost mode. The MO-TPE
settings match syftr's published optimizer, checked by an equivalence test.

## Setup

```bash
uv sync --extra dev
```

The bench reads API credentials from a `.env` file (symlink the framework's). It
needs `AZURE_API_KEY` and `AZURE_API_BASE`.

## Experiment 1: accuracy

The matrix is 3 datasets (HotpotQA, MuSiQue, MultiHop-RAG), 5 methods (`random`,
`motpe`, `motpe_warm`, `agentic_nokb_nodiag`, `agentic_score`), and 10 seeds, at 30
trials each, plus a `kb_greedy` reference at 10 seeds per dataset. Results land
under `experiment-1/<dataset>/`.

The scheduler `scripts/run_experiment1.py` runs each (method, seed) pair as its
own subprocess, keeps at most 2 running at once, starts `motpe_warm` only after
its matching `random` pair finishes, and writes each dataset's `Table_1.md` at
the end. Variance comes from the 10 seeds. `agentic_score` also reports its result
at 10 and 20 trials, for the sample-efficiency comparison.

```bash
# Preview the plan without running anything:
uv run python scripts/run_experiment1.py --dry-run --include-kb-greedy

# Run in the background (the full matrix takes about three to four days):
mkdir -p experiment-1/logs
setsid nohup uv run python scripts/run_experiment1.py --include-kb-greedy \
    > experiment-1/logs/nohup.out 2>&1 &

# Watch progress:
tail -f experiment-1/logs/scheduler.log
cat experiment-1/logs/STATUS.json
```

Each pair runs with `--resume`, and whether it is finished is read from files on
disk, not from exit codes. If the scheduler stops for any reason, run the same
command again and it skips finished work. Do not set `--workers` above 2, which
is the most the API endpoint handles reliably.

You can also run one dataset directly:

```bash
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml                    # full matrix
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m agentic_score   # one method
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml --resume           # continue after a stop
```

The KB and diagnosis ablations run on Hotpot only, after the headline, reusing
the same frozen exam:

```bash
uv run agentic-autorag-bench run --config configs/hotpot_ablation.yaml
```

## Experiment 2: cost and quality

This runs on UniDoc-Bench healthcare, about 230 biomedical PDFs and 20 images.
There is no gold answer set, so the optimizer generates its own exam and uses it
as both the tuning target and the score. Every method also minimizes per-query
LLM cost, which makes it a two-objective run. The matrix is 4 methods
(`agentic_cost`, `random`, `motpe`, `motpe_warm`) and 10 seeds, at 30 trials each.
Results land under `experiment-2/unidoc/`.

Unlike Experiment 1, this is a single command. The scheduler
`scripts/run_experiment2.py` first builds the shared setup (download and parse
the corpus, then generate and freeze the exam), then runs the 40 (method, seed)
pairs with at most 2 at once and the same `motpe_warm` after `random` gating, and
finally renders the figures and `hypervolume.json`.

```bash
# Preview the plan:
uv run python scripts/run_experiment2.py --dry-run

# Run in the background (the better part of a day, roughly 18 hours with 2 workers):
mkdir -p experiment-2/logs
setsid nohup uv run python scripts/run_experiment2.py --workers 2 \
    > experiment-2/logs/nohup.out 2>&1 &

# Watch progress:
tail -f experiment-2/logs/scheduler.log
cat experiment-2/logs/STATUS.json
```

The default config is `configs/unidoc_pareto.yaml`. Resume works as in Experiment
1: rerun the same command and finished pairs are skipped.

To rebuild the figures from a finished run without calling any API:

```bash
uv run agentic-autorag-bench pareto -c configs/unidoc_pareto.yaml --figure-only
```

This rewrites the figures in `experiment-2/unidoc/figures/` and `hypervolume.json`.
The two used in the paper are `pareto_cost_accuracy.png` (each method's
cost-accuracy frontier, best across seeds, with a min-max band where all seeds
cover) and `pareto_frontier_configs.png` (the same view with our frontier's actual
configurations labelled). The rest are supporting views:
`pareto_cost_accuracy_median.png` (the same frontier view but the line is the
median seed rather than the best, a typical-run rather than best-case reading),
`cost_and_embeddings.png` (search cost and embedding tokens per method, with seed
error bars), `pareto_agentic_cost.png` (a single agentic run's frontier),
`pareto_comparison.png` (one seed per method), `pareto_hypervolume.png`
(hypervolume over trials), and `pareto_median_and_hypervolume.png` (a wide
landscape pairing the median frontier and the hypervolume curve side by side).

The committed run keeps the per-pair results, the figures, and the frozen
`.shared_cache/exam.json` (the optimization target and score), dropping only the
large regeneratable rest of `.shared_cache/` (parsed corpus, embeddings, probe
indexes), so a fresh run reuses the committed exam rather than regenerating it.
Because RAG evaluation and LLM judging are not deterministic, a rerun reproduces
the finding (the shape of the frontier, the ordering of the methods, and the gain
from warm-starting) rather than the exact numbers.

## Output layout

```
experiment-1/hotpot/
  bench_metadata.json      dataset, methods, seeds, budget, optimizer version and commit
  .shared_cache/           parsed corpus, chunks, and embeddings, reused across methods and runs
  figures/                 cross-method figures and Table_1.md
  <method>/<seed>/
    benchmark_results.json held-out score of the best config
    best_config.yaml       the selected pipeline
    history.jsonl          per-trial config, score, tokens, and cost
    optimizer_meta.json    totals for tokens, cost, and wall-clock
```

Experiment 2 uses the same per-pair layout under
`experiment-2/unidoc/<method>/seed_<n>/`, with `history.jsonl` inside a
`details/` folder.

## Notes

- **Cost accounting.** Only the agentic methods generate an exam. The others
  reuse it. To keep the comparison fair, the cost tally excludes the tokens spent
  generating the exam.
- **Abstention.** MultiHop-RAG includes unanswerable questions, about 12% of the
  set. A system that abstains is scored correct, and one that answers anyway is
  scored wrong. These questions count toward answer accuracy but not retrieval
  metrics, since there are no documents to retrieve.
- **Provenance.** The optimizer lives at `../Agentic-AutoRAG` as a path dependency
  and keeps changing after the runs, so each result records the optimizer version
  and commit in `bench_metadata.json` — Experiment 1 under
  `experiment-1/<dataset>/` and Experiment 2 under `experiment-2/unidoc/`. Before
  the final runs, tag that commit, for example
  `git -C ../Agentic-AutoRAG tag v0.1.0-paper`.
- The earlier AutoRAG baseline was removed from the active matrix and is kept
  under `agentic_autorag_bench/_deprecated/`.
```

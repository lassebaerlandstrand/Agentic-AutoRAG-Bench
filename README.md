# Agentic AutoRAG — Benchmark Suite

Reproducibility artifact for the EMNLP paper. Runs a `(method × seed) × trials`
matrix on HotpotQA, MuSiQue, or MultiHop-RAG, then emits per-method figures and
a held-out `Table_1.md`.

## Methods

| Key                | Strategy                                                          | Per-trial signal                       | Seeded |
|--------------------|-------------------------------------------------------------------|----------------------------------------|--------|
| `agentic_score`    | Reasoning loop (ours), score-only objective (`cost_aware=False`)  | Open-ended LLM-judged exam, end-to-end | ✓      |
| `agentic_cost`     | Reasoning loop (ours), Pareto-aware objective (`cost_aware=True`) | Same exam                              | ✓      |
| `agentic_nokb`     | Ours, KB-off ablation (cold reasoning)                            | Same exam                              | ✓      |
| `agentic_nodiag`   | Ours, diagnosis-off ablation (registered; off the headline matrix)| Same exam                             | ✓      |
| `motpe`            | Optuna group-decomposed multivariate MO-TPE (= syftr's optimizer) | Same exam                              | ✓      |
| `motpe_warmstart`  | `motpe` seeded with the agent's frozen KB prior (cold proposer)   | Same exam                              | ✓      |
| `random`           | Random search                                                     | Same exam                              | ✓      |

All methods search the same `TrialConfig` space and are re-scored on the same
held-out exam, so the final-score column in `Table_1.md` is directly comparable
across rows (the YAHPO/HPOBench standard: fix the benchmark, swap the proposer).
`agentic_*` share the same `AgenticOptimizer` class — they differ only in the
`meta.cost_aware` flag and the `use_knowledge_base` / `use_diagnosis` ablation
toggles passed to the framework. `motpe` / `motpe_warmstart` share the
`MOTPESearch` class; `meta.cost_aware` switches it between single-objective
(accuracy) and two-objective (accuracy ↑, per-query cost ↓) mode, mirroring the
agentic flag. The MO-TPE sampler config (`multivariate, group, constant_liar`)
matches syftr's published optimizer — asserted by a behavioral-equivalence test.

The original Marker-Inc AutoRAG baseline (`autorag_our_exam`, `autorag_ragas`)
was dropped from the active matrix on 2026-05-27; the code is preserved
under `agentic_autorag_bench/_deprecated/autorag/` for possible resurrection
(see that directory's README for the reasoning and the steps to re-enable).

## Setup

```bash
uv sync --extra dev
# Bench reads .env (symlink the framework's) for AZURE_API_KEY, AZURE_API_BASE.
```

## Run

```bash
# Full matrix on one dataset
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml
uv run agentic-autorag-bench run --config configs/musique_paper.yaml
uv run agentic-autorag-bench run --config configs/multihop_rag_paper.yaml

# Specific methods — repeat -m for each
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m agentic_score
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m random -m motpe

# Resume a partial run within a method (skip the start-of-run reset)
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m agentic_score --no-clean
```

Method keys: `agentic_score`, `agentic_cost`, `agentic_nokb`, `agentic_nodiag`,
`motpe`, `motpe_warmstart`, `random`. Each one passed via `-m` must also be
declared in the config's `methods:` list.

Datasets are selected by config, not flag: each `configs/<dataset>_paper.yaml`
points at its own `output_root` (`results_hotpot/`, `results_paper_musique/`,
`results_multihop_rag/`) so the three runs don't collide.

**Scoped reset.** Each run wipes only the per-method dirs it's about to
write (and any matching `<method>@<k>` checkpoint dirs). Method dirs not
in this run, `.shared_cache/`, `bench_metadata.json`, and any user files
at `output_root` survive. The cross-method `figures/` directory is NOT
wiped at start-of-run — new matrix figures are rendered to a staging
directory and atomically swapped at the very end, so the previous run's
figures stay readable for the entire duration of a new run. Pass
`--no-clean` to skip the per-method reset entirely (useful for resuming
a partial run).

**Checkpoints.** Declare per-method early-stopping points in the bench
config to evaluate `history[:k]`'s best on the held-out QA as a sibling
`<method>@<k>/seed_<n>/` result directory:

```yaml
# configs/hotpot_paper.yaml
checkpoints:
  agentic_score: [10, 20]
  agentic_cost:  [10, 20]
```

Each declared `k < max_trials` adds one extra held-out evaluation per
seed. Lets the paper compare e.g. `agentic_score@20` vs. `motpe` at
the full 40-trial budget without paying for extra search trials. The
held-out judge caches per `(config_hash, question_id)`, so identical
configs across checkpoints incur no extra cost.

## Output layout

```
results_hotpot/                          # or results_paper_musique / results_multihop_rag
  bench_metadata.json                    # dataset + methods + seeds + max_trials + checkpoints
  filtered_questions.json                # held-out questions excluded across all runs (content-filter union)
  .shared_cache/                         # corpus + exam + embedding ingredients (reused across methods + runs)
    exam.json, exam_cost.json, cache_events.jsonl, ...
  figures/                               # matrix-level (cross-method); only updated at end-of-run via staging swap
    Table_1.md, score_per_trial.png, best_so_far.png, holdout_metrics.png,
    cost_breakdown.png, token_breakdown.png, appendix/
  <method>/                              # e.g. agentic_score, random, motpe
    figures/                             # per-method (across seeds)
    <seed_label>/
      figures/                           # per-seed (score_per_trial, cost_per_trial)
      benchmark_results.json             # held-out scoring on the best config
      best_config.yaml, recommended.yaml
      history.jsonl                      # per-trial config + score + tokens + eval_usd
      search_result.json                 # full SearchResult dump
      optimizer_meta.json                # roll-up: method, seed, totals, tokens, usd, wall-clock
      trial_cost_ledger.jsonl            # per-trial bucket delta (agent_proposal, rag_eval, judge, embedding_build)
      cache_events.jsonl                 # first-use cache-credit events (phase: exam_gen | trial)
      cost_breakdown.json                # framework-side run-total ledger (agentic only)
      bench_ledger.json                  # bench-side run-total ledger (non-agentic only)
      frontier.json, frontier/<trial>.yaml
      run.log
  <method>@<k>/                          # checkpoint sibling: e.g. agentic_score@10, agentic_score@20
    figures/                             # per-checkpoint (across seeds)
    <seed_label>/                        # same file layout as <method>/<seed_label>/; cumulative cost truncated to k trials
```

Figures emit progressively: per-seed after each `(method, seed)` finishes its
hold-out scoring; per-method after the method's seed loop closes; matrix-level
after the full run.

### Accounting model

Token accounting is **cache-aware** and uses the first-use-per-(method, seed)
rule: the first trial that touches a given cache key (embeddings, exam,
chunks) pays the deterministic token cost in its own `trial_cost_ledger.jsonl`
delta; later trials in the same run that hit the same key pay zero. This
makes `sum(trial.tokens) == framework_run_total` reconcile exactly while
preserving per-seed variance.

**Fairness rule (exam-gen exclusion).** Only `agentic_*` generates its own
exam; `random` / `motpe` reuse ours. Counting exam-gen cost would
penalise the method that creates exams for work the others don't do, so
the bench tally **excludes** exam-gen LLM and embedding tokens. The
framework still records its own exam-gen cost in the `exam_generation`
bucket for standalone (non-bench) runs; the bench's shared `.shared_cache/`
generates the exam without an active ledger, so the exam-cost sidecar is
zeros — correct under the rule. See `agentic_autorag_bench/types.py:TrialResult`
and `run.py:_make_metered_evaluator` for the implementation.

## Re-render figures from a committed tree

```bash
uv run agentic-autorag-bench analyze --results-dir results_hotpot/
```

Regenerates the matrix-level figures + `Table_1.md` without re-running the
matrix. Per-method and per-seed figures are already on disk from `run`.

## Deprecated baselines

The original Marker-Inc AutoRAG baseline lived under `methods/autorag/`
and was wired into this matrix until 2026-05-27. The driver, translator,
config-mirror, and setup script have been moved to
`agentic_autorag_bench/_deprecated/autorag/` and `scripts/_deprecated/`
respectively; the companion tests were deleted on decoupling and would
need to be rewritten on resurrection. See
`agentic_autorag_bench/_deprecated/README.md` for the reasoning, the
original setup steps, and the wiring needed to re-enable.

## Framework dependency

Editable path dep on `../Agentic-AutoRAG` (see `[tool.uv.sources]`). The
optimizer keeps evolving on `main` after the paper benchmarks run, so
reproducibility rests on provenance, not a frozen checkout: every run stamps the
optimizer's package version and git commit into `<output_root>/bench_metadata.json`
(`run._optimizer_provenance`), so each results directory self-documents the exact
code that produced it.

Once, right before the final benchmark runs, tag the optimizer at the commit you
run — a venue-neutral name, e.g. `git -C ../Agentic-AutoRAG tag v0.1.0-paper` —
so the stamped `describe` resolves to that tag. Bump the suffix (`v0.1.1-paper`)
if you re-run for a resubmission.

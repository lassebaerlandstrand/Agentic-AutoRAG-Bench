# Agentic AutoRAG — Benchmark Suite

Reproducibility artifact for the EMNLP paper. Runs a `(method × seed) × trials`
matrix on HotpotQA, MuSiQue, or MultiHop-RAG, then emits per-method figures and
a held-out `Table_1.md`.

## Methods

| Key                | Strategy                                                          | Per-trial signal                          | Seeded |
|--------------------|-------------------------------------------------------------------|-------------------------------------------|--------|
| `agentic_score`    | Reasoning loop (ours), score-only objective (`cost_aware=False`)  | Open-ended LLM-judged exam, end-to-end    | ✓      |
| `agentic_cost`     | Reasoning loop (ours), Pareto-aware objective (`cost_aware=True`) | Same exam                                 | ✓      |
| `random`           | Random search                                                     | Same exam                                 | ✓      |
| `bayesian`         | Optuna TPE                                                        | Same exam                                 | ✓      |
| `autorag_our_exam` | Marker-Inc AutoRAG, our exam questions as QA                      | AutoRAG native per-node metrics (greedy)  | —      |
| `autorag_ragas`    | Marker-Inc AutoRAG, RAGAS-bootstrapped QA                         | AutoRAG native per-node metrics (greedy)  | —      |

All six search the same `TrialConfig` space and are re-scored on the same
held-out exam, so the final-score column in `Table_1.md` is directly
comparable across rows. `agentic_score` and `agentic_cost` share the same
`AgenticOptimizer` class — they differ only in the `meta.cost_aware` flag
passed to the framework, which switches the Proposer between score-maximising
and Pareto-aware stances.

## Setup

```bash
uv sync --extra dev
# AutoRAG runs in its own venv (pins numpy<2). The script also patches
# chroma's batched insert for our 19k-doc corpus. The bench auto-discovers
# the resulting .autorag-venv/ — no env var needed for the default layout.
# Set AUTORAG_PYTHON only if the venv lives somewhere else.
bash scripts/setup_autorag_venv.sh
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
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m random -m bayesian

# Resume a partial run within a method (skip the start-of-run reset)
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m agentic_score --no-clean

# Dev iteration (1 seed, 3 trials, tiny search space — minutes per method)
uv run agentic-autorag-bench run --config configs/hotpot_dev.yaml
```

Method keys: `agentic_score`, `agentic_cost`, `random`, `bayesian`,
`autorag_our_exam`, `autorag_ragas`. Each one passed via `-m` must also be
declared in the config's `methods:` list.

Datasets are selected by config, not flag: each `configs/<dataset>_paper.yaml`
points at its own `output_root` (`results_paper/`, `results_paper_musique/`,
`results_paper_multihop_rag/`) so the three runs don't collide.

**Scoped reset.** Each run wipes only the method dirs it's about to write,
plus the cross-method `figures/` dir (regenerated at end-of-run). Untargeted
method dirs, `.shared_cache/`, and any user files at `output_root` survive,
so `-m agentic_score` composes with previous `random`/`bayesian`/`autorag_*`
results. Back up manually with `cp -r results_paper results_paper_<date>`
before a run if you want a snapshot. Pass `--no-clean` to skip the reset
entirely.

## Output layout

```
results_hotpot/                          # or results_musique / results_multihop_rag
  bench_metadata.json                   # dataset + methods + seeds + max_trials
  filtered_questions.json               # held-out questions excluded across all runs (content-filter union)
  .shared_cache/                        # corpus + exam + embedding ingredients (reused across methods)
    exam.json, exam_cost.json, cache_events.jsonl, ...
  figures/                              # matrix-level (cross-method)
    Table_1.md, score_per_trial.png, best_so_far.png, holdout_metrics.png,
    cost_breakdown.png, token_breakdown.png, appendix/
  <method>/
    figures/                            # per-method (across seeds)
    <seed_label>/
      figures/                          # per-seed (score_per_trial, cost_per_trial)
      benchmark_results.json            # held-out scoring on the best config
      best_config.yaml, recommended.yaml
      history.jsonl                     # per-trial config + score + tokens + eval_usd
      search_result.json                # full SearchResult dump
      optimizer_meta.json               # roll-up: method, seed, totals, tokens, usd, wall-clock
      trial_cost_ledger.jsonl           # per-trial bucket delta (agent_proposal, rag_eval, judge, embedding_build)
      cache_events.jsonl                # first-use cache-credit events (phase: exam_gen | trial)
      cost_breakdown.json               # framework-side run-total ledger (agentic only)
      bench_ledger.json                 # bench-side run-total ledger (non-agentic only)
      frontier.json, frontier_report.md, frontier/<trial>.yaml
      run.log
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

**Fairness rule (exam-gen exclusion).** Only `agentic_*` and `autorag_ragas`
generate their own exams; `random` / `bayesian` / `autorag_our_exam` reuse
ours. Counting exam-gen cost would penalise the two methods that create
exams for work the others don't do, so the bench tally **excludes** exam-gen
LLM and embedding tokens. The framework still records its own exam-gen cost
in the `exam_generation` bucket for standalone (non-bench) runs; the bench's
shared `.shared_cache/` generates the exam without an active ledger, so the
exam-cost sidecar is zeros — correct under the rule. See
`agentic_autorag_bench/types.py:TrialResult` and `run.py:_make_metered_evaluator`
for the implementation.

## Re-render figures from a committed tree

```bash
uv run agentic-autorag-bench analyze --results-dir results_paper/
```

Regenerates the matrix-level figures + `Table_1.md` without re-running the
matrix. Per-method and per-seed figures are already on disk from `run`.

## AutoRAG baseline — caveats

Schema is split into lexical / semantic / hybrid retrieval nodes (we emit all
three with `hybrid_cc.weight=1.0` so a `vector_only` space is a semantic
pass-through); chunking is frozen across all six methods
(`strategy=recursive, chunk_size=512, overlap=0`) because AutoRAG's evaluator
has no chunking node; Azure LLMs use `llm: openai` against Azure's `/openai/v1`
shim (`openailike` drops `is_chat_model` and misroutes chat models to
`/completions`); Bedrock LLMs use `llm: bedrock_converse` (registered into
`autorag.generator_models` by `scripts/autorag_patches.py`) because AutoRAG
0.3's bundled `llama-index-llms-bedrock` is the deprecated package that
hard-restricts `model` to a pre-2024 registry, and the search space's
Llama 3.1 / Nova 2 / Claude Haiku 4.5 entries all fail there. `BedrockConverse`
needs `region_name` (boto3 reads `AWS_REGION` / `AWS_DEFAULT_REGION`, not
the litellm-convention `AWS_REGION_NAME`), so we plumb `${AWS_REGION_NAME}`
explicitly into the generator modules. Long form in the paper appendix and
`methods/autorag/`.

## Framework dependency

Path dep on `../Agentic-AutoRAG` during development; switched to a git
submodule pinned at `v0.1.0-emnlp2026` before paper submission.

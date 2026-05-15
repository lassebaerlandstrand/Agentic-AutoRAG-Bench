# Agentic AutoRAG — Benchmark Suite

Reproducibility artifact for the EMNLP paper. Runs a `(method × seed) × trials`
matrix on HotpotQA, then emits per-method figures and a held-out `Table_1.md`.

## Methods

| Key             | Strategy                                     | Per-trial signal                          | Seeded |
|-----------------|----------------------------------------------|-------------------------------------------|--------|
| `agentic`       | Reasoning loop (ours)                        | Open-ended LLM-judged exam, end-to-end    | ✓      |
| `random`        | Random search                                | Same exam                                 | ✓      |
| `bayesian`      | Optuna TPE                                   | Same exam                                 | ✓      |
| `autorag_mcq`   | Marker-Inc AutoRAG, our exam questions as QA | AutoRAG native per-node metrics (greedy)  | —      |
| `autorag_ragas` | Marker-Inc AutoRAG, RAGAS-bootstrapped QA    | AutoRAG native per-node metrics (greedy)  | —      |

All five search the same `TrialConfig` space and are re-scored on the same
held-out exam, so the final-score column in `Table_1.md` is directly
comparable across rows.

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
# Full matrix
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml

# Specific methods — repeat -m for each
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m agentic
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m random -m bayesian

# Resume a partial run within a method (skip the start-of-run reset)
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m agentic --no-clean

# Dev iteration (1 seed, 3 trials, tiny search space — minutes per method)
uv run agentic-autorag-bench run --config configs/hotpot_dev.yaml
```

Method keys: `agentic`, `random`, `bayesian`, `autorag_mcq`, `autorag_ragas`.
Each one passed via `-m` must also be declared in the config's `methods:` list.

**Scoped reset.** Each run wipes only the method dirs it's about to write,
plus the cross-method `figures/` dir (regenerated at end-of-run). Untargeted
method dirs, `.shared_cache/`, and any user files at `output_root` survive,
so `-m agentic` composes with previous `random`/`bayesian`/`autorag_*` results.
Back up manually with `cp -r results_paper results_paper_<date>` before a run
if you want a snapshot. Pass `--no-clean` to skip the reset entirely.

## Output layout

```
results_paper/
  figures/                              # matrix-level (cross-method)
    Table_1.md, score_per_trial.png, best_so_far.png,
    holdout_metrics.png, efficiency.png, cost_breakdown.png
  <method>/
    figures/                            # per-method (across seeds)
    <seed_label>/
      figures/                          # per-seed
      history.jsonl, benchmark_results.json, ...
```

Figures emit progressively: per-seed after each `(method, seed)` finishes its
hold-out scoring; per-method after the method's seed loop closes; matrix-level
after the full run.

## Re-render figures from a committed tree

```bash
uv run agentic-autorag-bench analyze --results-dir results_paper/
```

Regenerates the matrix-level figures + `Table_1.md` without re-running the
matrix. Per-method and per-seed figures are already on disk from `run`.

## AutoRAG baseline — caveats

Schema is split into lexical / semantic / hybrid retrieval nodes (we emit all
three with `hybrid_cc.weight=1.0` so a `vector_only` space is a semantic
pass-through); chunking is frozen across all 5 methods
(`strategy=recursive, chunk_size=256, overlap=0`) because AutoRAG's evaluator
has no chunking node; LLM uses `llm: openai` against Azure's `/openai/v1`
shim (`openailike` drops `is_chat_model` and misroutes chat models to
`/completions`). Long form in the paper appendix and `methods/autorag/`.

## Framework dependency

Path dep on `../Agentic-AutoRAG` during development; switched to a git
submodule pinned at `v0.1.0-emnlp2026` before paper submission.

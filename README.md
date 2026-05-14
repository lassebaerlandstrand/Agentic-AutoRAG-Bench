# Agentic AutoRAG — Benchmark Suite

Reproducibility artifact for the EMNLP submission comparing
[Agentic AutoRAG](https://github.com/lassebaerlandstrand/Agentic-AutoRAG) against:

| Row | Method                                 | Per-trial optimization signal                                       | Search        |
|-----|----------------------------------------|---------------------------------------------------------------------|---------------|
| 1   | Agentic AutoRAG (ours)                 | Framework's open-ended exam (LLM-judged), end-to-end                | reasoning loop |
| 2   | Random search                          | Framework's open-ended exam (LLM-judged), end-to-end                | sequential     |
| 3   | Bayesian (Optuna TPE)                  | Framework's open-ended exam (LLM-judged), end-to-end                | sequential     |
| 4   | Marker-Inc AutoRAG (RAGAS bootstrap QA)| AutoRAG native per-node metrics (retrieval_f1, ROUGE/BLEU)          | greedy per node|
| 5   | Marker-Inc AutoRAG (our exam as QA)    | AutoRAG native per-node metrics on our exam's retrieval_gt + answers | greedy per node|

**Optimization-signal semantics.** Rows 1–3 are scored end-to-end on every
trial by the same LLM-judged exam evaluator. Rows 4–5 use AutoRAG's *native*
greedy per-node optimization (each node picks its own winner using its own
metric: retrieval_f1/recall/precision for retrieval/reranker/query-expansion,
ROUGE+BLEU for prompt_maker/generator — see
[AutoRAG strategies](https://marker-inc-korea.github.io/AutoRAG/optimization/strategies.html)).
The difference between row 4 and row 5 is only the QA dataset AutoRAG
optimizes against: row 4 uses RAGAS-bootstrapped QA, row 5 uses our exam's
questions with `source_doc_ids` as `retrieval_gt` and
`canonical_answer + answer_variants` as `generation_gt`. Neither AutoRAG row
uses our LLM judge as the per-node signal — that would violate AutoRAG's
design — but every winning AutoRAG config is **re-scored** on the same
held-out exam evaluator as rows 1–3, so the final-score column is directly
comparable across all rows.

All five rows search the same `TrialConfig` configuration space and the
held-out evaluation uses the same HotpotQA QA via Agentic AutoRAG's
`benchmark-evaluate`.

## Layout

```
configs/                    paper-mode YAML (search space, budget, seeds, judge)
agentic_autorag_bench/
  types.py                  Optimizer protocol + shared dataclasses
  space.py                  SearchSpaceSpec ↔ framework SearchSpace
  methods/                  random, bayesian, agentic, autorag
  benchmarks/               hotpot_qa.py (wraps benchmark-prepare/evaluate)
  run.py                    (method × seed) matrix orchestrator
  plots.py                  auto-figures: per-seed → per-method → matrix
  analyze.py                bootstrap CIs + Table_1.md (re-render entry point)
scripts/setup_autorag_venv.sh
results_paper/              run output. Per-seed data + figures/, plus
                            per-method <method>/figures/, plus matrix
                            figures/Table_1.md, score_per_trial.png, ...
tests/
```

## Figures

Every matrix run emits figures progressively, so a partial run is still
inspectable. Three nesting levels share the same canonical names
(``score_per_trial.png``, ``best_so_far.png``, ``holdout_metrics.png``,
``efficiency.png``, ``cost_breakdown.png``, ``cost_per_trial.png``) so a
reader can navigate down the tree and recognise the same view at each
level.

```
results_paper/
  figures/                              # matrix-level (cross-method)
    Table_1.md
    score_per_trial.png                 # per-trial mean ± std, one line / method
    best_so_far.png                     # best-so-far, sequential methods only
    holdout_metrics.png                 # EM/F1/Judge grouped bars per method
    efficiency.png                      # score-vs-cost, score-vs-wallclock
    cost_breakdown.png                  # optimizer vs trial $ stack per method
  <method>/
    figures/                            # method-level (across seeds)
      score_per_trial.png               # one line per seed, mean ± std band
      best_so_far.png
      holdout_metrics.png               # per-seed bars
    <seed_label>/
      figures/                          # per-seed (single run)
        score_per_trial.png             # raw + best-so-far + hold-out marker
        cost_per_trial.png              # per-trial + cumulative
      history.jsonl, benchmark_results.json, ...
```

Per-seed figures are written as soon as a single ``(method, seed)`` finishes
its hold-out scoring; per-method figures right after that method's seed loop
closes; matrix figures after the full run (post union-exclusion).

**Scoped last-run-wins.** Each ``run`` resets only the method dirs it's
about to run (plus the cross-method ``figures/`` dir, which gets
regenerated from whatever is in the tree). So ``-m agentic`` wipes
``agentic/`` and ``figures/`` and leaves ``random/``, ``bayesian/``,
``autorag_*/``, ``.shared_cache/``, and any user files untouched — partial
runs compose with previous results for the other methods. The
cross-method figures the new run produces will reflect "new agentic + old
random + old bayesian", which is usually what you want. Back up the
directory manually (``cp -r results_paper results_paper_<date>``) if you
want to snapshot before a run; pass ``--no-clean`` to skip the reset
entirely (useful for resuming a partial run within a method).

## Setup

```bash
uv sync --extra dev
# AutoRAG runs in its own venv (it pins numpy<2, conflicting with our base
# deps). The script installs AutoRAG[gpu] + drops a chroma-batching patch.
bash scripts/setup_autorag_venv.sh
export AUTORAG_PYTHON=$(pwd)/.autorag-venv/bin/python
```

The bench reads `.env` (symlinked to the framework's own `.env`) for
`AZURE_API_KEY`, `AZURE_API_BASE`. All paper configs use azure/ LLM ids.

## Run the matrix

```bash
# Dev iteration: 1 seed, 3 trials, 50-question exam, 100-question hold-out
# on the full ~19k-doc hotpot_val_2000 corpus. Smaller search space than the
# paper run (1 embed, 2 LLMs, vector_only); ~tens of minutes per method.
uv run agentic-autorag-bench run --config configs/hotpot_dev.yaml
# Figures land in results_dev/figures/, results_dev/<method>/figures/, and
# results_dev/<method>/<seed_label>/figures/ as the run progresses.

# Paper run: 3 seeds, 10 trials, 2000 questions — hours.
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml
# Same: results_paper/figures/ + per-method + per-seed.
```

## Reproducing the paper

The published `results_paper/` directory contains every winning config and
its held-out scoring; the matrix-level figures and ``Table_1.md`` are
emitted in-place as the run finishes. To regenerate them from a committed
tree without re-running the (~5-day, ~$300) matrix:

```bash
uv run agentic-autorag-bench analyze --results-dir results_paper/
# Writes results_paper/figures/{Table_1.md, score_per_trial.png, ...}
```

The matrix-level figures (in ``results_paper/figures/``):

- `Table_1.md` — per-method EM / F1 / Judge with bootstrap CIs, cost, wall.
- `score_per_trial.png` — per-trial exam score, mean ± std across seeds,
  one line per sequential method. The "is the optimizer actually working?"
  view.
- `best_so_far.png` — best-so-far per trial, mean ± std across seeds. Same
  data as the legacy ``figure_trajectory.png`` (which is also written for
  compat).
- `holdout_metrics.png` — grouped bars of EM / F1 / Judge per method.
- `efficiency.png` — score vs. cost and score vs. wall-clock (one point per
  method, 95% bootstrap CI on the score axis).
- `cost_breakdown.png` — stacked bar of optimizer-side vs trial-side cost.

To re-run from scratch (compute-heavy):

```bash
rm -rf results_paper/
uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml
```

## AutoRAG baseline — known limitations (paper appendix)

- **Schema fidelity**: AutoRAG v0.3 split retrieval into three node types
  (lexical / semantic / hybrid). The reranker downstream needs the un-suffixed
  retrieval columns, which only `hybrid_retrieval` produces. We therefore
  always emit all three retrieval nodes; for a `vector_only`-only search
  space we pin `hybrid_cc.weight=1.0` (semantic-only) so the hybrid node acts
  as a pass-through of the semantic retriever. AutoRAG's `hybrid_cc.weight`
  is the *semantic* weight (`weight=1.0` → semantic-only, `weight=0.0` →
  BM25-only — see
  [autorag/nodes/hybridretrieval/hybrid_cc.py](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/autorag/nodes/hybridretrieval/hybrid_cc.py)),
  identical convention to our `hybrid_alpha`. The translator maps weight
  through directly. This is documented in the per-run
  `translation_notes.json`.
- **Chunking is frozen across all 5 methods (Path B).** AutoRAG's
  [`chunk` phase](https://marker-inc-korea.github.io/AutoRAG/data_creation/chunk/chunk.html)
  can sweep `chunk_method` and `chunk_size` (as a Cartesian product of YAML
  lists, expanded via
  [`make_combinations`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/autorag/utils/util.py))
  and emits one parquet per combination, but: (a) `chunk_overlap` is a
  scalar — AutoRAG's chunker does not enumerate it; (b) there is no
  selection step — `summary.csv` only records execution time, not
  retrieval quality; (c) there is no chunk node inside `autorag evaluate`,
  so the optimizer treats the chosen corpus as fixed. AutoRAG's own FAQ
  flags chunking optimization as
  [not yet implemented inside the evaluator](https://medium.com/@autorag/faq-tips-tricks-f6b09a989d5e).
  Implementing a fair outer chunking loop for all 5 methods (Path A) would
  multiply our compute budget by `K` (number of chunkings) without
  resolving (a). To keep the comparison defensible, we instead pin every
  method to the same `(strategy=recursive, chunk_size=256, overlap=0)`
  triple. The paper claim narrows to "given the same chunking, our
  optimizer beats the alternatives"; an outer chunking sweep is named as
  future work in the limitations section.
- **LLM provider**: AutoRAG's `llm: openailike` drops `is_chat_model` through
  `pop_params` (it's a Pydantic class attribute, not an init arg), routing
  chat models to `/completions` (HTTP 400). We use `llm: openai` with
  `api_base=${AZURE_API_BASE}/openai/v1` instead — model-name-based chat
  detection then routes correctly.
- **Azure content filter**: a single offending question (e.g. a HotpotQA
  Wikipedia paragraph that triggers `ResponsibleAIPolicyViolation` for
  violence) crashes the entire AutoRAG pipeline (no per-question retry).
  When this happens the matrix runner catches the exception and the AutoRAG
  row is omitted; rerunning against a less restrictive deployment or
  pre-filtering the QA set is the workaround.
- **Chroma SQLite batch cap**: `Chroma.add_embedding` hands the whole corpus
  to chroma in one `.add()` call, which exceeds the SQLite backend's 5461-row
  batch limit on our 19k-doc corpus (`Batch size of N is greater than max
  batch size of 5461`). `scripts/setup_autorag_venv.sh` drops
  `scripts/autorag_patches.py` into the autorag venv's site-packages via a
  `.pth` line so every interpreter invocation chunks the insert. Re-run the
  setup script if you rebuild the venv.
- **RAGAS QA bootstrap**: AutoRAG v0.3.x exposes the QA-generation chain at
  `autorag.data.qa.*` (not `autorag.data.beta.*` as in earlier builds) and
  uses the `Corpus().sample().make_retrieval_gt_contents().batch_apply()`
  pipeline. The bench's `qa_ragas.py` routes the bootstrap LLM through Azure
  with the same `<AZURE_API_BASE>/openai/v1` shim used elsewhere.

## Framework dependency

Pinned via `pyproject.toml`. During development this is a path dep on
`../Agentic-AutoRAG`. Before paper submission this switches to a git
submodule pinned to `v0.1.0-emnlp2026`.

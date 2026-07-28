# Agentic AutoRAG Benchmark Suite

Benchmark and reproduction code for the Agentic AutoRAG paper. It compares our
reasoning-agent optimizer against random search and MO-TPE (the optimizer syftr
uses) on two experiments:

- **Accuracy experiment** — each optimizer tunes a RAG pipeline on HotpotQA,
  MuSiQue, and MultiHop-RAG, scored on a held-out gold slice of each dataset.
- **Pareto experiment** — the optimizers tune for accuracy and per-query cost
  together on UniDoc-Bench healthcare, tracing a cost-quality frontier.

**The completed runs are committed.** Every table and figure in the paper can be
regenerated from this repository in under a minute, with no API keys, no GPU,
and no network. That is the [Reproduce the paper](#reproduce-the-paper) path
below, and it is the one that actually verifies the paper's numbers.
[Rerun from scratch](#rerun-from-scratch) is a separate, multi-day, four-provider
undertaking described afterwards.

## Contents

| Path | What it is |
|---|---|
| `agentic_autorag_bench/` | The bench package: matrix orchestrator, search methods, analysis. |
| `configs/` | One `*_paper.yaml` per Accuracy dataset, `unidoc_pareto.yaml` for the Pareto run; each pairs with a `*_project.yaml` holding the search space and model roles. |
| `scripts/` | The paper's figure/table generators and the two long-run schedulers. |
| `experiment-1/<dataset>/` | Accuracy-experiment results: 3 datasets x 5 methods x 10 seeds, plus the `kb_greedy` reference and the @10/@20 checkpoints. |
| `experiment-2/unidoc/` | Pareto-experiment results: 4 methods x 10 seeds, plus the frozen exam. |
| `benchmark_data/<dataset>/` | The frozen validation exam, the stratified splits, and corpus provenance. |
| `tests/` | 263 tests. All LLM calls are mocked; the suite needs no network. |

The optimizer itself lives in the sibling repository `../Agentic-AutoRAG` and is
consumed here as an editable path dependency, so the two directories must sit
next to each other.

## Setup

```bash
uv sync --extra dev
uv run pytest          # 263 passed, no network needed
```

That is all the [Reproduce the paper](#reproduce-the-paper) path requires.

## Reproduce the paper

Each command reads the committed results and rewrites the artifact in place.
Run them from the repository root.

| Paper artifact | Command | Output |
|---|---|---|
| **Table 2** (`tab:holdout`) — held-out accuracy | `uv run python scripts/paper_figures.py` | `experiment-1/figures_paper/table1_answer_quality.tex` |
| **Figure 2** (`fig:best-so-far`) — best-so-far curves | `uv run python scripts/best_so_far_figure.py` | `experiment-1/figures_paper/best_so_far_3panel.pdf` |
| **Figure 3** (`fig:pareto`) — cost-accuracy frontier | `uv run agentic-autorag-bench pareto -c configs/unidoc_pareto.yaml --figure-only` | `experiment-2/unidoc/figures/pareto_cost_accuracy_median.pdf` |
| **Appendix figure** (`fig:cost-emb`) — search cost + embedding tokens | `uv run python scripts/paper_figures.py` | `experiment-1/figures_paper/cost_and_embeddings.pdf` |
| **Section 5.2** significance numbers | `uv run python scripts/exp2_significance.py` | `experiment-2/unidoc/significance.md` (also printed) |
| Per-dataset `Table_1.md` + matrix figures | `uv run agentic-autorag-bench analyze --results-dir experiment-1/hotpot` | `experiment-1/hotpot/figures/` |

The `table1_answer_quality.tex` / `Table_1.md` filenames predate the paper's
final float ordering — they hold the paper's **Table 2**. `paper_figures.py`
also emits `score_per_trial_3panel.pdf` and `answer_quality.pdf`, and
`pareto --figure-only` rewrites the whole `experiment-2/unidoc/figures/` set and
`hypervolume.json`; the paper uses the rows above. Repeat the `analyze` command
with `--results-dir experiment-1/musique` and `experiment-1/multihop` for the
other two datasets. Whole table: about half a minute.

Two small paper tables are transcribed from committed data rather than emitted
by a script:

```bash
# Table 1 (`tab:corpora`), first three rows (the UniDoc row is measured on the
# downloaded corpus, see "Rerun from scratch")
uv run python -c "
import json,pathlib
for d in ['hotpot_val_2000','musique_val_2417','multihop_rag_val']:
    m=json.loads(pathlib.Path(f'benchmark_data/{d}/metadata.json').read_text())
    print(f\"{m['name']:<14} {m['corpus_doc_count']:>6} docs  {m['corpus_total_words']/1e6:.2f}M words  {m['corpus_avg_words_per_doc']:.0f} w/doc\")"

# Supplement table `tab:unidoc-type-counts`
uv run python -c "
import json,collections
ex=json.load(open('experiment-2/unidoc/.shared_cache/exam.json'))
print(collections.Counter(q['reasoning_type'] for q in ex), 'total', len(ex))"
```

### Verifying the evaluation set is the one that produced the results

The frozen exam records a SHA-256 of the optimization pool it was drawn from, so
the committed exam and the committed splits can be checked against each other:

```bash
uv run python -c "
import hashlib,json,pathlib
for d in ['hotpot_val_2000','musique_val_2417','multihop_rag_val']:
    pool=json.loads(pathlib.Path(f'benchmark_data/{d}/splits/optimization_qa.json').read_text())
    got=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()
    want=json.loads(pathlib.Path(f'benchmark_data/{d}/validation_exam_provenance.json').read_text())['qa_sha256']
    print(d, 'MATCH' if got==want else 'MISMATCH')"
```

## Methods

| Key | Description |
|-----|-------------|
| `agentic_score` | Our reasoning-loop optimizer, tuning for accuracy. The Accuracy experiment's method under test. |
| `agentic_cost` | Our optimizer in cost-aware mode, tuning accuracy and cost together. The Pareto experiment's method under test. |
| `agentic_nokb_nodiag` | Our optimizer stripped to score-only: no knowledge base, no diagnosis, compact config-to-score history. The paper's "Agentic (no KB/diag)" ablation, run as a rival baseline. |
| `random` | Random search. Also the transfer source that seeds `motpe_warm`. |
| `motpe` | Optuna group-multivariate MO-TPE, cold. The syftr-class statistical baseline. |
| `motpe_warm` | The same MO-TPE, warm-started from this run's paired `random` trials as a free, uncounted prior. |
| `agentic_nokb`, `agentic_nodiag` | Single-component ablations. Implemented and runnable via `configs/hotpot_ablation.yaml`, but **not run for the paper**. |
| `qlognehvi` | A GP-BO reference (Barker et al.). Cited in the paper, not run. Needs `uv add ax-platform`. |

All methods search the same pipeline space and are scored the same way, so their
results compare directly. The agentic methods share one optimizer class and
differ only by config flags. The two MO-TPE methods share another class, with a
flag selecting accuracy-only or accuracy-and-cost mode; their settings match
syftr's published optimizer, checked by an equivalence test.

## Output layout

Both experiments use the same per-pair layout.

```
experiment-1/hotpot/                     # = output_root from the config
  bench_metadata.json                    dataset, hf_revision, methods, seeds, budget,
                                         and the optimizer version + commit that ran it
  filtered_questions.json                questions excluded from every method's denominator
  .shared_cache/                         parsed corpus, chunks, embeddings (regeneratable, not committed)
  figures/                               cross-method figures + Table_1.md
  figures_paper/                         the cross-dataset paper figures + table1_answer_quality.tex
  <method>/seed_<n>/
    benchmark_results.json               held-out score of the best config, with per-question rows
    best_config.yaml                     the selected pipeline
    recommended.yaml                     the optimizer's recommended pick
    search_result.json                   search-level summary
    optimizer_meta.json                  totals: trials, tokens, USD, wall-clock
    wall_clock.json                      wall-clock for the non-agentic methods
    frontier/trial_NN.yaml               Pareto-frontier configs
    figures/                             per-seed score/cost curves
    details/history.jsonl                per-trial config, score, tokens, cost, and agent diagnosis
    details/trial_cost_ledger.jsonl      per-call cost ledger
    details/cost_breakdown.json          optimizer-vs-trial cost split
    run.log                              full console log, including every agent prompt
```

`<method>@<k>/seed_<n>/` directories (e.g. `agentic_score@10/`) hold the
early-stopping checkpoints: the best config within the method's first *k* trials,
scored on the same held-out slice. They back the paper's sample-efficiency claim.

Experiment 2 is the same, under `experiment-2/unidoc/<method>/seed_<n>/`, with
two differences: there is no `benchmark_results.json` (UniDoc has no gold answer
set, so trials are scored on the optimizer's own frozen exam), and the exam
itself is committed at `experiment-2/unidoc/.shared_cache/exam.json`.

## Rerun from scratch

This reproduces the *findings*, not the numbers — see
[What is and is not reproducible](#what-is-and-is-not-reproducible). Budget
several days and roughly \$1,000 in API spend.

### Credentials

A full rerun spans four providers. `.env.example` documents every variable;
copy it to `.env` and fill in the four groups below.

**A `.env` file on disk is not enough.** LiteLLM reads credentials from the
process environment, and neither the bench CLI nor `uv run` loads `.env`
automatically. Export it into the shell before every run — either

```bash
export UV_ENV_FILE=.env        # uv loads it for every `uv run` in this shell
```

or

```bash
set -a; . ./.env; set +a       # plain shell export
```

Whichever you use, apply it in the same shell that launches the schedulers, so
the `(method, seed)` subprocesses inherit it.

| Provider | Prefix | Used for | Variables |
|---|---|---|---|
| AWS Bedrock | `bedrock/` | Examiner (`minimax.minimax-m2.5`), optimizer agent (`moonshotai.kimi-k2.5`), and most generators | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` |
| Azure OpenAI | `azure/` | GPT-family generators | `AZURE_API_KEY`, `AZURE_API_BASE` |
| Azure AI Foundry | `azure_ai/` | The judge (`DeepSeek-V3.2`) | `AZURE_AI_API_KEY`, `AZURE_AI_API_BASE` |
| Google Vertex AI | `vertex_ai/` | Gemini generators | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS` |

**Azure deployment names are per-subscription.** Each `configs/*_project.yaml`
carries a `model_aliases:` block mapping the config's model strings to the
deployment names on our subscription:

```yaml
model_aliases:
  azure/gpt-5-nano: azure/gpt-5-nano-1
  ...
```

Re-point those to your own deployments, or the run fails partway through. Verify
the whole search space is reachable before committing to a multi-day run:

```bash
uv run python scripts/preflight_search_space.py configs/hotpot_paper_project.yaml --env .env
```

(This script is the one place that *does* read a `.env` directly; without
`--env` it looks for `../Agentic-AutoRAG/.env`.)

Rebuilding the frozen validation exams additionally uses Google AI Studio
(`gemini/`, `GEMINI_API_KEY`) as the span extractor — but the exams are
committed, so a rerun does not need it.

### Rebuild the corpora

The corpora are large and regeneratable, so they are not committed; the pinned
HuggingFace revision is. `run` prepares them automatically from the
`benchmark.hf_revision` in each config, but you can do it explicitly:

```bash
uv run agentic-autorag benchmark-prepare hotpot_qa \
  -o ./benchmark_data/hotpot_val_2000 --split validation \
  --sample-size 2000 --seed 42 --hf-revision 1908d6afbbead072334abe2965f91bd2709910ab

uv run agentic-autorag benchmark-prepare musique \
  -o ./benchmark_data/musique_val_2417 --split validation \
  --sample-size 2417 --seed 42 --hf-revision c8f4f8c9465fb69d31a8eae894c3fd509c4ca321

# MultiHop-RAG ships HuggingFace *configs*, not splits: --split names the config,
# and the default `validation` errors out.
uv run agentic-autorag benchmark-prepare multihop_rag \
  -o ./benchmark_data/multihop_rag_val --split MultiHopRAG \
  --sample-size 2000 --seed 42 --hf-revision 71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82
```

Each writes a fresh `metadata.json`. Check `corpus_doc_count` and
`corpus_total_words` against the committed values — those are the numbers in the
paper's corpus table, and a mismatch means the corpus differs from ours. The
committed `splits/` and `validation_exam.json` are left alone, so the evaluation
set stays fixed regardless.

The UniDoc corpus (230 healthcare PDFs plus 20 page images, ~336 MB) downloads
automatically on the first `pareto` run from `Salesforce/UniDoc-Bench`. That
download is *not* revision-pinned; the exam that defines every Pareto score is
committed, so a corpus drift would change retrieval rather than the target.

### Accuracy experiment

3 datasets x 5 methods x 10 seeds at 30 trials each, plus a `kb_greedy`
reference. `scripts/run_experiment1.py` runs each (method, seed) pair as its own
subprocess, keeps at most 2 running, starts `motpe_warm` only after its paired
`random` finishes, and writes each dataset's `Table_1.md` at the end.

```bash
uv run python scripts/run_experiment1.py --dry-run --include-kb-greedy   # preview the DAG

mkdir -p experiment-1/logs
setsid nohup uv run python scripts/run_experiment1.py --include-kb-greedy \
    > experiment-1/logs/nohup.out 2>&1 &

tail -f experiment-1/logs/scheduler.log
cat experiment-1/logs/STATUS.json
```

Or drive one dataset directly:

```bash
uv run agentic-autorag-bench run -c configs/hotpot_paper.yaml                    # full matrix
uv run agentic-autorag-bench run -c configs/hotpot_paper.yaml -m agentic_score   # one method
uv run agentic-autorag-bench run -c configs/hotpot_paper.yaml --resume           # continue after a stop
uv run agentic-autorag-bench kb-greedy -c configs/hotpot_paper.yaml --seed 1     # the no-search reference
```

### Pareto experiment

4 methods x 10 seeds at 30 trials on UniDoc-Bench healthcare. There is no gold
answer set, so the optimizer generates its own exam and uses it as both tuning
target and score, and every method also minimizes per-query LLM cost.
`scripts/run_experiment2.py` builds the shared setup (download and parse the
corpus, then load the committed frozen exam), runs the 40 pairs with the same
2-worker and `motpe_warm`-after-`random` rules, and renders the figures and
`hypervolume.json`.

```bash
uv run python scripts/run_experiment2.py --dry-run

mkdir -p experiment-2/logs
setsid nohup uv run python scripts/run_experiment2.py --workers 2 \
    > experiment-2/logs/nohup.out 2>&1 &
```

### Ablations

The paper's ablation, `agentic_nokb_nodiag`, is part of the headline matrix
above. The single-component decomposition was not run for the paper; to run it:

```bash
uv run agentic-autorag-bench run -c configs/hotpot_ablation.yaml
```

It reuses the headline's project YAML and frozen exam, and writes to its own
`experiment-1/hotpot_ablation/`.

### Rerun ergonomics

- **Completion is read from disk, not exit codes.** A scheduler that stops for
  any reason resumes correctly by re-running the same command; finished pairs
  are skipped. This also means the *committed* results read as already done — a
  genuine rerun needs a fresh `output_root`. Copy the config and change one line:
  ```bash
  sed 's|^output_root: .*|output_root: ./rerun-1/hotpot|' \
      configs/hotpot_paper.yaml > configs/hotpot_rerun.yaml
  uv run agentic-autorag-bench run -c configs/hotpot_rerun.yaml
  ```
  `configs/*_project.yaml` is referenced relative to the config file, so a copy
  in `configs/` needs no other edit.
- **`--clean` refuses to destroy finished work.** A fresh `run` defaults to
  cleaning the method dirs it is about to write, but it will not delete dirs that
  already hold completed hold-out results. Pass `--resume` to continue, or
  `--force` to deliberately restart.
- **Do not raise `--workers` above 2.** That is the most the API endpoints handle
  reliably; higher settings produce throttling errors that corrupt a run's cost
  and coverage accounting rather than just slowing it down.
- **GPU.** Index building and embedding run on the local GPU when one is present.
  A rerun works on CPU but is considerably slower.

### Cost and wall-clock

Measured from the committed runs (`optimizer_meta.json`), excluding
exam generation:

| | Search USD | Compute-hours | Wall-clock at 2 workers |
|---|---|---|---|
| Accuracy, HotpotQA | \$209 | 52 | ~26 h |
| Accuracy, MuSiQue | \$313 | 54 | ~27 h |
| Accuracy, MultiHop-RAG | \$384 | 58 | ~29 h |
| Held-out scoring (240 evals, all 3) | \$125 | — | — |
| Pareto, UniDoc | \$134 | 35 | ~18 h |
| **Total** | **\$1,165** | **199** | **~100 h** |

## What is and is not reproducible

Fixed by construction, because they are committed: the validation exams, the
stratified held-out slices, the UniDoc exam, the search space, the seeds, and
the pinned dataset revisions. Sampling variation is therefore not a source of
run-to-run difference.

Not fixed: hosted LLM generation and LLM judging are nondeterministic, and the
hosted models themselves are versioned by the provider and change over time. A
rerun reproduces the paper's *findings* — the ordering of the methods, the
sample-efficiency gap, the warm-start gain, the shape of the cost-quality
frontier — not its exact numbers.

## Notes

- **Cost accounting.** Only the agentic methods generate an exam; the others
  reuse it. To keep the comparison fair, the cost tally excludes the tokens spent
  generating the exam.
- **Abstention.** MultiHop-RAG includes unanswerable questions, about 12% of the
  set. A system that abstains is scored correct, one that answers anyway is
  scored wrong. These count toward answer accuracy but not retrieval metrics,
  since there are no documents to retrieve.
- **Cross-method exclusion.** If any method's hold-out eval hits a provider
  content filter on a question, that question is dropped from *every* method's
  denominator, so all methods are scored on the same question set. The union is
  recorded in `filtered_questions.json`.
- **Provenance.** The optimizer at `../Agentic-AutoRAG` keeps changing after the
  runs, so each results tree records the optimizer version and commit in
  `bench_metadata.json`. Before final runs, tag that commit, e.g.
  `git -C ../Agentic-AutoRAG tag v0.1.0-paper`.
- The earlier AutoRAG baseline was removed from the active matrix and is kept
  under `agentic_autorag_bench/_deprecated/`.

## License

MIT — see `LICENSE`.

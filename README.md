# Agentic AutoRAG — Benchmark Suite

Reproducibility artifact for the EMNLP submission comparing
[Agentic AutoRAG](https://github.com/lassebaerlandstrand/Agentic-AutoRAG) against:

| Row | Method                                | Optimization signal     | Sequential? |
|-----|---------------------------------------|--------------------------|-------------|
| 1   | Agentic AutoRAG (ours)                | Framework's open-ended exam | yes      |
| 2   | Random search                         | Framework's open-ended exam | yes      |
| 3   | Bayesian (Optuna TPE)                 | Framework's open-ended exam | yes      |
| 4   | Marker-Inc AutoRAG (RAGAS native)     | AutoRAG-bootstrapped QA     | greedy/no |
| 5   | Marker-Inc AutoRAG (greedy + our exam)| Framework's open-ended exam | greedy/no |

All five rows search the same `TrialConfig` configuration space and are scored
by the same held-out HotpotQA QA via Agentic AutoRAG's `benchmark-evaluate`.

## Layout

```
configs/                    paper-mode YAML (search space, budget, seeds, judge)
autorag_bench/
  types.py                  Optimizer protocol + shared dataclasses
  space.py                  SearchSpaceSpec ↔ framework SearchSpace
  methods/                  random, bayesian, agentic, autorag
  benchmarks/               hotpot_qa.py (wraps benchmark-prepare/evaluate)
  run.py                    (method × seed) matrix orchestrator
  analyze.py                bootstrap CIs, paper tables, trajectory figures
scripts/setup_autorag_venv.sh
results/                    committed: best_config.yaml + history.jsonl + benchmark_results.json
paper_artifacts/            committed: Table_1.tex, figure_trajectory.pdf
tests/
```

## Setup

```bash
uv sync --extra dev
# AutoRAG runs in its own venv (it pins numpy<2, conflicting with our base
# deps). The script defaults to API-mode; add [gpu] for local HF embedders.
bash scripts/setup_autorag_venv.sh
.autorag-venv/bin/pip install "AutoRAG[gpu]"   # needed for HF embedders / rerankers
export AUTORAG_PYTHON=$(pwd)/.autorag-venv/bin/python
```

The bench reads `.env` (symlinked to the framework's own `.env`) for
`AZURE_API_KEY`, `AZURE_API_BASE`. All paper configs use azure/ LLM ids.

## Run the matrix

```bash
# Dev iteration: 1 seed, 3 trials, 50 questions — finishes in minutes.
uv run autorag-bench run --config configs/hotpot_dev.yaml
uv run autorag-bench analyze --results-dir results_dev/ --output dev_artifacts/

# Paper run: 3 seeds, 10 trials, 2000 questions — hours.
uv run autorag-bench run --config configs/hotpot_paper.yaml
uv run autorag-bench analyze --results-dir results/ --output paper_artifacts/
```

## Reproducing the paper

The published `results/` directory contains every winning config and its
held-out scoring; `analyze.py` regenerates `paper_artifacts/Table_1.tex` and
`figure_trajectory.pdf` from those committed artifacts in seconds, without
re-running the (~5-day, ~$300) matrix.

To re-run from scratch (compute-heavy):

```bash
rm -rf results/
uv run autorag-bench run --config configs/hotpot_paper.yaml
uv run autorag-bench analyze
```

## Framework dependency

Pinned via `pyproject.toml`. During development this is a path dep on
`../Agentic-AutoRAG`. Before paper submission this switches to a git
submodule pinned to `v0.1.0-emnlp2026`.

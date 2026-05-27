# Deprecated baselines (frozen)

This directory holds code that is no longer wired into the active bench
matrix but is preserved so it can be resurrected without rewriting from
scratch. Treat everything here as **frozen as of 2026-05-27**: it is not
maintained, tests are excluded from the default `pytest` run, and the
active code paths (`run.py`, `cli.py`, `configs/*.yaml`) do not import any
of it.

## What's here

### `autorag/` — Marker-Inc AutoRAG (the *original* AutoRAG)

A two-method baseline (`autorag_our_exam`, `autorag_ragas`) that ran the
original AutoRAG via a `subprocess` into its own venv, then translated the
best-config YAML back into our `TrialConfig` for rescoring on the shared
evaluator.

**Why it was decoupled.** Including AutoRAG in the paper matrix forced the
search space down to its lowest common denominator (every numeric dim had
to be `DiscreteValues`, chunking frozen, small generator pool), which made
the comparison unfair to our framework once the paper moved to a wider
HotpotQA space and 40-trial budgets. Dropping AutoRAG from the active
matrix lets the other methods run against a search space large enough to
distinguish them.

### Companion files

- `scripts/_deprecated/setup_autorag_venv.sh` — provisions `.autorag-venv/`
  with the original AutoRAG (numpy<2 isolation).
- `scripts/_deprecated/autorag_patches.py` — runtime patches the AutoRAG
  venv loads at startup (Chroma batching, BedrockConverse LLM
  registration).

The matching `test_autorag_*.py` and `test_qa_prescreen.py` suites were
deleted when AutoRAG was decoupled — restoring the baseline means writing
fresh tests against the current `agentic_autorag_bench.types` /
`agentic_autorag.config.models` surface. Don't try to revive the original
tests from git history; they referenced an older schema.

## How to resurrect

If you decide to re-enable AutoRAG for a follow-up matrix:

1. Restore the active-code wiring in `agentic_autorag_bench/run.py`:
   - Add `from agentic_autorag_bench._deprecated.autorag.driver import AutoRAGOptimizer, resolve_autorag_python` to the imports.
   - Put `"autorag_ragas"` and `"autorag_our_exam"` back into `DETERMINISTIC_METHODS`.
   - Restore the `autorag_*` branch in `_build_optimizer()`.
   - Restore the `resolve_autorag_python()` preflight before `if clean:`.
2. Add the two method names back to `configs/<benchmark>.yaml::methods`.
3. Restore the autorag entries in `agentic_autorag_bench/plots.py::METHOD_ORDER`
   and `agentic_autorag_bench/analyze.py::METHOD_ORDER`.
4. Run `scripts/_deprecated/setup_autorag_venv.sh` to provision `.autorag-venv/`.
5. Write fresh tests for the resurrected baseline (the original
   `test_autorag_*.py` suites were deleted on decoupling — see "What's
   here").
6. Audit the search-space translator against any new `SearchSpace` field
   that has been added since the freeze date. The translator only knows
   about fields it shipped with — new fields silently get default values
   on the AutoRAG side, which breaks fairness.

The framework's `agentic_autorag.config.models::SearchSpace` is the source
of truth for what AutoRAG must mirror; the translator is at
`autorag/translator.py`.

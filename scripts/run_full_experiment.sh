#!/usr/bin/env bash
# Run the full paper experiment — the 3-dataset accuracy matrix (HotpotQA,
# MuSiQue, MultiHop-RAG) plus the UniDoc cost-vs-accuracy Pareto demo — as ONE
# resumable command.
#
#   ./scripts/run_full_experiment.sh            # run everything (auto-resume)
#   ./scripts/run_full_experiment.sh --smoke    # tiny 2-trial/2-seed dry run of the whole pipeline
#   ./scripts/run_full_experiment.sh --no-pareto
#   ./scripts/run_full_experiment.sh --fresh    # DESTRUCTIVE: wipe + restart every stage
#   ./scripts/run_full_experiment.sh --only hotpot,musique
#
# RESUME MODEL (the important bit for a multi-day run):
#   * Each dataset runs in its OWN subprocess (`agentic-autorag-bench run`), so a
#     crash in one cannot corrupt another's state or leak global LLM/runtime state.
#   * Re-running the SAME command after a crash is always safe and never wipes
#     finished work: a stage that has already started (its output_root has a
#     bench_metadata.json) is relaunched with `--resume`; a stage that never
#     started runs fresh. Completed (method, seed) pairs are skipped instantly by
#     the in-harness completion guard, so a resume only pays for unfinished work.
#   * `--fresh` is the ONLY way to wipe and restart, and it is never the default.
#
# Stages run sequentially and independently; if one fails the script records it,
# continues to the next, and exits non-zero so you know to re-run (which resumes).
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SMOKE=0
RUN_PARETO=1
FRESH=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --no-pareto) RUN_PARETO=0; shift ;;
    --fresh) FRESH=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  # name|bench-config  (smoke configs: 2 trials × 2 seeds × all 5 methods)
  ACCURACY_STAGES=(
    "hotpot|configs/smoke_hotpot.yaml"
    "musique|configs/smoke_musique.yaml"
    "multihop|configs/smoke_multihop_rag.yaml"
  )
  PARETO_CONFIG=""        # no smoke pareto config; skip the pareto stage under --smoke
else
  ACCURACY_STAGES=(
    "hotpot|configs/hotpot_paper.yaml"
    "musique|configs/musique_paper.yaml"
    "multihop|configs/multihop_rag_paper.yaml"
  )
  PARETO_CONFIG="configs/unidoc_pareto.yaml"
fi

LOG_DIR="$REPO_ROOT/experiment_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Pretty status accumulators
declare -a RESULTS=()

_output_root_of() {  # parse `output_root:` out of a bench config
  grep -E '^output_root:' "$1" | head -1 | awk '{print $2}'
}

_stage_selected() {  # honor --only name1,name2
  [[ -z "$ONLY" ]] && return 0
  [[ ",$ONLY," == *",$1,"* ]] && return 0
  return 1
}

run_accuracy_stage() {
  local name="$1" cfg="$2"
  local out_root resume_flag log
  out_root="$(_output_root_of "$cfg")"
  log="$LOG_DIR/${name}_${STAMP}.log"

  if [[ "$FRESH" -eq 1 ]]; then
    echo ">>> [$name] FRESH: wiping $out_root" | tee -a "$log"
    rm -rf "$out_root"
    resume_flag=""
  elif [[ -f "$out_root/bench_metadata.json" ]]; then
    echo ">>> [$name] prior run detected at $out_root → --resume" | tee -a "$log"
    resume_flag="--resume"
  else
    echo ">>> [$name] no prior state → fresh start" | tee -a "$log"
    resume_flag=""
  fi

  echo ">>> [$name] $(date -Is)  uv run agentic-autorag-bench run -c $cfg $resume_flag" | tee -a "$log"
  # shellcheck disable=SC2086
  if uv run agentic-autorag-bench run -c "$cfg" $resume_flag >>"$log" 2>&1; then
    echo ">>> [$name] DONE" | tee -a "$log"
    RESULTS+=("$name: OK  ($log)")
    return 0
  else
    echo ">>> [$name] FAILED (see $log) — re-run this script to resume" | tee -a "$log"
    RESULTS+=("$name: FAILED  ($log)")
    return 1
  fi
}

run_pareto_stage() {
  local cfg="$1"
  local out_root resume_flag log
  out_root="$(_output_root_of "$cfg")"
  log="$LOG_DIR/pareto_${STAMP}.log"

  # The NEW Pareto run (agentic_cost vs motpe_warmstart) leaves these markers;
  # an older/incompatible results_unidoc/ (e.g. a single-method run) has neither.
  local has_new_marker=0
  if [[ -f "$out_root/hypervolume.json" || -d "$out_root/motpe_warmstart" ]]; then
    has_new_marker=1
  fi

  if [[ "$FRESH" -eq 1 ]]; then
    echo ">>> [pareto] FRESH: wiping $out_root" | tee -a "$log"
    rm -rf "$out_root"
    resume_flag=""
  elif [[ "$has_new_marker" -eq 1 ]]; then
    echo ">>> [pareto] prior new-Pareto run detected → --resume" | tee -a "$log"
    resume_flag="--resume"
  elif [[ -d "$out_root" && -n "$(ls -A "$out_root" 2>/dev/null)" ]]; then
    # Non-empty but no new-Pareto marker → stale/incompatible data. Don't
    # silently resume onto it or auto-delete it; make the user choose.
    echo ">>> [pareto] SKIPPED: $out_root has stale/incompatible data (no new-Pareto" | tee -a "$log"
    echo ">>>          markers). Pass --fresh to wipe + rerun, or 'rm -rf $out_root' first." | tee -a "$log"
    RESULTS+=("pareto: SKIPPED-STALE  ($log)")
    return 0
  else
    resume_flag=""
  fi

  echo ">>> [pareto] $(date -Is)  uv run agentic-autorag-bench pareto -c $cfg $resume_flag" | tee -a "$log"
  # shellcheck disable=SC2086
  if uv run agentic-autorag-bench pareto -c "$cfg" $resume_flag >>"$log" 2>&1; then
    echo ">>> [pareto] DONE" | tee -a "$log"
    RESULTS+=("pareto: OK  ($log)")
    return 0
  else
    echo ">>> [pareto] FAILED (see $log) — re-run this script to resume" | tee -a "$log"
    RESULTS+=("pareto: FAILED  ($log)")
    return 1
  fi
}

FAILED=0
echo "============================================================"
echo " Agentic-AutoRAG full experiment  (smoke=$SMOKE pareto=$RUN_PARETO fresh=$FRESH only='${ONLY:-all}')"
echo " logs: $LOG_DIR"
echo "============================================================"

for stage in "${ACCURACY_STAGES[@]}"; do
  name="${stage%%|*}"
  cfg="${stage##*|}"
  _stage_selected "$name" || { echo ">>> [$name] skipped (--only)"; continue; }
  run_accuracy_stage "$name" "$cfg" || FAILED=1
done

if [[ "$RUN_PARETO" -eq 1 && -n "$PARETO_CONFIG" ]]; then
  if _stage_selected "pareto"; then
    run_pareto_stage "$PARETO_CONFIG" || FAILED=1
  fi
fi

echo
echo "==================== SUMMARY ==============================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "==========================================================="
if [[ "$FAILED" -ne 0 ]]; then
  echo "One or more stages failed. Re-run this script (without --fresh) to resume them."
  exit 1
fi
echo "All stages complete."

#!/usr/bin/env bash
# Launch one test/example dataset run (agentic full + one MO-TPE, seed 1, 40
# trials) with the gemini-3.5-flash examiner. Usage: ./run_testrun.sh <ds>
#   ds = musique | multihop_rag
# Datasets have disjoint output_root/.shared_cache, so two may run concurrently
# (aggregate DeepSeek judge concurrency 2x10=20 < ~50 ceiling; GPU fits).
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
unset VIRTUAL_ENV
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Quiet google.auth's "No project ID could be determined" spam for vertex_ai
# (cosmetic; vertex calls already work via VERTEXAI_PROJECT/ADC).
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-${VERTEXAI_PROJECT:-}}"
ds="${1:?usage: run_testrun.sh <musique|multihop_rag>}"
mkdir -p experiment_logs
log="experiment_logs/testrun_${ds}.log"
echo ">>> [$ds] START $(date -Is)  pid=$$  args=${*:2}" | tee -a "$log"
uv run agentic-autorag-bench run -c "configs/${ds}_testrun.yaml" "${@:2}" >> "$log" 2>&1
rc=$?
echo ">>> [$ds] END   $(date -Is)  exit=$rc" | tee -a "$log"
exit $rc

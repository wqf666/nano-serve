#!/usr/bin/env bash
# =============================================================================
# check_tp_env.sh — Launch the T0 Tensor Parallel environment check
# =============================================================================
# Usage:
#     bash scripts/check_tp_env.sh [NUM_GPUS]
#
# Default: 2 GPUs. Requires torchrun and at least NUM_GPUS visible CUDA devices.
# =============================================================================
set -euo pipefail

NUM_GPUS="${1:-2}"

echo "============================================================"
echo "NanoServe TP Environment Check"
echo "Launching torchrun with nproc_per_node=${NUM_GPUS}"
echo "============================================================"

# Verify we have enough GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
if [ "${AVAILABLE_GPUS}" -lt "${NUM_GPUS}" ]; then
    echo "ERROR: Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} are available."
    echo "       Check nvidia-smi output and CUDA_VISIBLE_DEVICES."
    exit 1
fi

echo "Detected ${AVAILABLE_GPUS} GPU(s). Proceeding with ${NUM_GPUS}."
echo ""

# Clean previous results
rm -rf tp_env_check_results

# Launch via torchrun
torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    -m nanovllm.offline.tp_env_check

EXIT_CODE=$?

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "============================================================"
    echo "SUCCESS: All TP environment checks passed."
    echo "Results written to tp_env_check_results/"
    echo "============================================================"
else
    echo "============================================================"
    echo "FAILURE: TP environment check failed (exit code ${EXIT_CODE})."
    echo "Check tp_env_check_results/ for per-rank details."
    echo "============================================================"
fi

exit ${EXIT_CODE}

#!/usr/bin/env bash
# =============================================================================
# run_offline_tp_bench.sh — Benchmark tensor parallel offline generation
# =============================================================================
# Usage:
#     bash scripts/run_offline_tp_bench.sh [MODEL_NAME] [MAX_NEW_TOKENS]
#
# Runs generation in both TP=1 (single GPU) and TP=2 (torchrun) modes,
# then prints a side-by-side comparison.
#
# Prerequisites:
#   - 2x CUDA GPUs visible
#   - nanovllm package on PYTHONPATH
#   - transformers, torch installed
# =============================================================================
set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"
MAX_NEW_TOKENS="${2:-64}"
OUTPUT_DIR="tp_bench_results_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "NanoServe Tensor Parallel Benchmark"
echo "============================================================"
echo "Model:          ${MODEL_NAME}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Verify GPU availability
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
echo "Detected ${AVAILABLE_GPUS} GPU(s):"
nvidia-smi --list-gpus 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""

# ------------------------------------------------------------------
# Phase 1: TP=1 (single GPU baseline)
# ------------------------------------------------------------------
echo "============================================================"
echo "Phase 1: TP=1 (Single GPU Baseline)"
echo "============================================================"

TP1_START=$(date +%s%N)

python -m nanovllm.offline.tp_generate \
    --model-name "${MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --benchmark \
    --output-dir "${OUTPUT_DIR}"

TP1_END=$(date +%s%N)
TP1_ELAPSED_MS=$(( (TP1_END - TP1_START) / 1000000 ))

echo ""
echo "TP=1 completed in ${TP1_ELAPSED_MS} ms"
echo ""

# ------------------------------------------------------------------
# Phase 2: TP=2 (tensor parallel)
# ------------------------------------------------------------------
if [ "${AVAILABLE_GPUS}" -ge 2 ]; then
    echo "============================================================"
    echo "Phase 2: TP=2 (Tensor Parallel, 2 GPUs)"
    echo "============================================================"

    TP2_START=$(date +%s%N)

    torchrun \
        --standalone \
        --nproc_per_node=2 \
        -m nanovllm.offline.tp_generate \
        --model-name "${MODEL_NAME}" \
        --tensor-parallel-size 2 \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --benchmark \
        --output-dir "${OUTPUT_DIR}"

    TP2_END=$(date +%s%N)
    TP2_ELAPSED_MS=$(( (TP2_END - TP2_START) / 1000000 ))

    echo ""
    echo "TP=2 completed in ${TP2_ELAPSED_MS} ms"
else
    echo "============================================================"
    echo "Phase 2: SKIPPED (only ${AVAILABLE_GPUS} GPU available, need 2)"
    echo "============================================================"
    TP2_ELAPSED_MS="N/A"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Benchmark Summary"
echo "============================================================"
echo "Model:             ${MODEL_NAME}"
echo "Max new tokens:    ${MAX_NEW_TOKENS}"
echo "TP=1 wall time:    ${TP1_ELAPSED_MS} ms"
echo "TP=2 wall time:    ${TP2_ELAPSED_MS} ms"
echo ""
echo "Detailed results in: ${OUTPUT_DIR}/"
echo "  - result_tp1.json"
if [ "${AVAILABLE_GPUS}" -ge 2 ]; then
    echo "  - result_tp2.json"
fi
echo "============================================================"

# ------------------------------------------------------------------
# GPU memory stats
# ------------------------------------------------------------------
echo ""
echo "GPU Memory Usage (post-benchmark):"
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free \
    --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

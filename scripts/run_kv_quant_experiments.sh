#!/usr/bin/env bash
# ===========================================================================
# run_kv_quant_experiments.sh  —  Q3 Experiment Script
# ===========================================================================
# Compare fp16, int8, fp8 KV cache quantization on NanoServe workloads.
#
# Outputs
# -------
# results/kv_quant_results.json  — raw metrics per (workload, dtype)
#
# Prerequisites
# -------------
# - NanoServe repo at /root/autodl-tmp/nanoserve/repos/nanoserve
# - Python 3.10+ with torch, transformers, flash-attn
# - GPU with ≥ 24 GB VRAM (4090 / A5000 / A100)
# ===========================================================================

set -euo pipefail

# ---- Config ---------------------------------------------------------------
REPO_DIR="${NANOSERVE_REPO:-/root/autodl-tmp/nanoserve/repos/nanoserve}"
RESULTS_DIR="${REPO_DIR}/results"
PYTHON="${PYTHON:-python3}"

# Workload definitions
#   long-doc:         single request, seq_len=8192
#   mixed-enterprise: batch of 8 requests, mixed lengths (512–4096)
WORKLOADS=("long-doc" "mixed-enterprise")
DTYPES=("fp16" "int8" "fp8")

# Model
MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"

# ---- Setup ----------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"

echo "============================================="
echo " NanoServe KV Cache Quantization Experiments"
echo "============================================="
echo " Repo   : ${REPO_DIR}"
echo " Model  : ${MODEL_ID}"
echo " Results: ${RESULTS_DIR}"
echo " Date   : $(date -Iseconds)"
echo " GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================="
echo ""

# ---- Quick correctness check ----------------------------------------------
echo "[Step 0] Running correctness verification on random tensors ..."
cd "${REPO_DIR}"
${PYTHON} -c "
import json, sys
sys.path.insert(0, '.')
from nanovllm.kv_cache.quantized_kv_cache import verify_quantization_correctness

results = verify_quantization_correctness(num_blocks=10)
print(json.dumps(results, indent=2))

# Sanity checks
for dtype in ['int8', 'fp8']:
    stats = results[dtype]
    if stats['max_abs_error'] > 0.5:
        print(f'WARNING: {dtype} max_abs_error={stats[\"max_abs_error\"]:.4f} exceeds threshold', file=sys.stderr)
        sys.exit(1)
print('Correctness check PASSED.')
" 2>&1 | tee "${RESULTS_DIR}/correctness_check.log"
echo ""

# ---- Run experiments ------------------------------------------------------
RESULTS_JSON="${RESULTS_DIR}/kv_quant_results.json"

# Initialize JSON
echo '[]' > "${RESULTS_JSON}"

for workload in "${WORKLOADS[@]}"; do
    for dtype in "${DTYPES[@]}"; do
        echo "---------------------------------------------------"
        echo " Workload: ${workload}  |  KV dtype: ${dtype}"
        echo "---------------------------------------------------"

        # Build the command
        # The experiment runner is expected to:
        #   1. Launch the NanoServe server with --kv-cache-dtype <dtype>
        #   2. Send the workload requests
        #   3. Measure peak KV memory via torch.cuda.max_memory_allocated()
        #   4. Measure throughput (tokens/sec)
        #   5. Compare outputs to fp16 baseline for consistency
        #   6. Print JSON metrics to stdout
        #
        # If the full server harness is not yet wired up, we fall back to
        # the standalone benchmark that exercises QuantizedKVCache directly.

        CMD=(
            ${PYTHON} -m nanovllm.benchmarks.kv_quant_bench
            --workload "${workload}"
            --kv-cache-dtype "${dtype}"
            --model "${MODEL_ID}"
            --output-json "${RESULTS_DIR}/tmp_${workload}_${dtype}.json"
        )

        # Set workload-specific params
        if [ "${workload}" == "long-doc" ]; then
            CMD+=(--seq-len 8192 --batch-size 1 --num-requests 1)
        elif [ "${workload}" == "mixed-enterprise" ]; then
            CMD+=(--seq-len 4096 --batch-size 8 --num-requests 8)
        fi

        echo " CMD: ${CMD[*]}"
        echo ""

        # Run (allow failure so we can collect partial results)
        if "${CMD[@]}" 2>&1; then
            echo "  -> OK"
        else
            echo "  -> FAILED (exit $?), recording partial result"
            # Write a stub result so the plot script doesn't crash
            cat > "${RESULTS_DIR}/tmp_${workload}_${dtype}.json" <<STUB
{
    "workload": "${workload}",
    "dtype": "${dtype}",
    "peak_kv_memory_mb": null,
    "throughput_tokens_per_sec": null,
    "output_match_rate": null,
    "error": "benchmark run failed"
}
STUB
        fi

        # Append to results array
        if [ -f "${RESULTS_DIR}/tmp_${workload}_${dtype}.json" ]; then
            ${PYTHON} -c "
import json, sys
with open('${RESULTS_JSON}') as f:
    results = json.load(f)
with open('${RESULTS_DIR}/tmp_${workload}_${dtype}.json') as f:
    entry = json.load(f)
results.append(entry)
with open('${RESULTS_JSON}', 'w') as f:
    json.dump(results, f, indent=2)
"
        fi

        echo ""
    done
done

# ---- Summary --------------------------------------------------------------
echo "============================================="
echo " All experiments complete."
echo " Results written to: ${RESULTS_JSON}"
echo "============================================="
echo ""

cat "${RESULTS_JSON}"

echo ""
echo "To plot results, run:"
echo "  python benchmark/plot_kv_quant_results.py ${RESULTS_JSON}"
echo ""

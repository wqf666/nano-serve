#!/bin/bash
# Run all enterprise offline experiments (single GPU: RTX 4090D)
# Usage: bash scripts/run_enterprise_offline_experiments.sh

set -e
MODEL=~/huggingface/Qwen3-0.6B
REPO=/root/autodl-tmp/nanoserve/repos/nanoserve
RESULTS=/root/autodl-tmp/nanoserve/results/offline_enterprise
PY=/root/autodl-tmp/nanoserve/envs/nanoserve/bin/python

export HF_HUB_DISABLE_XET=1
cd $REPO

echo "=== Generating workloads ==="
mkdir -p /root/autodl-tmp/nanoserve/datasets/offline_jobs
$PY -c "
import json, sys; sys.path.insert(0, '.')
from benchmark.offline_workloads import generate_offline_workload
for wl in ['mixed-enterprise', 'shared-prefix', 'short-batch', 'long-doc']:
    items = generate_offline_workload(wl, 128, seed=42)
    path = f'/root/autodl-tmp/nanoserve/datasets/offline_jobs/{wl}_128.jsonl'
    with open(path, 'w') as f:
        for it in items:
            f.write(json.dumps({'id': it.id, 'prompt': it.prompt, 'max_tokens': it.max_tokens, 'prefix_key': it.prefix_key}) + '\n')
    print(f'  Generated {len(items)} items for {wl}')
"

# ---- Experiment 1: Planner comparison on mixed-enterprise (128 reqs) ----
echo ""
echo "=== Experiment 1: Planner comparison (mixed-enterprise, 128 reqs) ==="
for PLANNER in fcfs length_sorted length_bucket token_budget length_bucket_token_budget; do
    echo "  Running planner=$PLANNER ..."
    $PY benchmark/offline_bench.py \
        --model $MODEL --workload mixed-enterprise --num-requests 128 \
        --batch-size 16 --max-tokens 256 --planner $PLANNER \
        --max-batch-tokens 4096 --preserve-output-order --enforce-eager --no-tqdm \
        --save-result $RESULTS/planner_${PLANNER}_mixed_128.json
done

# ---- Experiment 2: Planner comparison on shared-prefix (128 reqs) ----
echo ""
echo "=== Experiment 2: Planner comparison (shared-prefix, 128 reqs) ==="
for PLANNER in fcfs prefix_grouped prefix_then_length_bucket_token_budget; do
    echo "  Running planner=$PLANNER ..."
    $PY benchmark/offline_bench.py \
        --model $MODEL --workload shared-prefix --num-requests 128 \
        --batch-size 16 --max-tokens 256 --planner $PLANNER \
        --max-batch-tokens 4096 --preserve-output-order --enforce-eager --no-tqdm \
        --save-result $RESULTS/planner_${PLANNER}_shared_128.json
done

# ---- Experiment 3: Batch size sweep (mixed-enterprise, length_bucket_token_budget) ----
echo ""
echo "=== Experiment 3: Batch size sweep ==="
for BS in 8 16 32; do
    echo "  Running batch_size=$BS ..."
    $PY benchmark/offline_bench.py \
        --model $MODEL --workload mixed-enterprise --num-requests 128 \
        --batch-size $BS --max-tokens 256 \
        --planner length_bucket_token_budget \
        --max-batch-tokens 4096 --preserve-output-order --enforce-eager --no-tqdm \
        --save-result $RESULTS/bs${BS}_mixed_128.json
done

# ---- Experiment 4: Token budget sweep ----
echo ""
echo "=== Experiment 4: Token budget sweep ==="
for MBT in 2048 4096 8192; do
    echo "  Running max_batch_tokens=$MBT ..."
    $PY benchmark/offline_bench.py \
        --model $MODEL --workload mixed-enterprise --num-requests 128 \
        --batch-size 16 --max-tokens 256 \
        --planner length_bucket_token_budget \
        --max-batch-tokens $MBT --preserve-output-order --enforce-eager --no-tqdm \
        --save-result $RESULTS/mbt${MBT}_mixed_128.json
done

echo ""
echo "=== All experiments complete ==="
echo "Run: $PY benchmark/plot_offline_results.py --input-dir $RESULTS --output-dir $RESULTS/figures"

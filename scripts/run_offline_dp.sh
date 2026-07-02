#!/bin/bash
# Run offline data parallel inference on multiple GPUs.
# Usage: bash scripts/run_offline_dp.sh [NUM_GPUS]

NUM_GPUS=${1:-1}
MODEL=~/huggingface/Qwen3-0.6B
REPO=/root/autodl-tmp/nanoserve/repos/nanoserve
OUTPUT=/root/autodl-tmp/nanoserve/results/offline_dp

export HF_HUB_DISABLE_XET=1
PYTHON=/root/autodl-tmp/nanoserve/envs/nanoserve/bin/python

# Build GPU list
GPUS="0"
if [ "$NUM_GPUS" -gt 1 ]; then
    GPUS="0"
    for i in $(seq 1 $((NUM_GPUS - 1))); do
        GPUS="$GPUS,$i"
    done
fi

echo "Running offline DP with $NUM_GPUS GPU(s): $GPUS"

cd $REPO

# Generate workload if not exists
WORKLOAD_JSONL=/root/autodl-tmp/nanoserve/datasets/offline_jobs/mixed_enterprise_256.jsonl
if [ ! -f "$WORKLOAD_JSONL" ]; then
    echo "Generating workload..."
    mkdir -p $(dirname $WORKLOAD_JSONL)
    $PYTHON -c "
import json, sys
sys.path.insert(0, '$REPO')
from benchmark.offline_workloads import generate_offline_workload
items = generate_offline_workload('mixed-enterprise', 256, seed=42)
with open('$WORKLOAD_JSONL', 'w') as f:
    for item in items:
        f.write(json.dumps({'id': item.id, 'prompt': item.prompt, 'max_tokens': item.max_tokens, 'prefix_key': item.prefix_key}) + '\n')
print(f'Generated {len(items)} items')
"
fi

# Run 1-GPU baseline
RUN_DIR=$OUTPUT/run_1gpu
echo "=== 1 GPU baseline ==="
$PYTHON -m nanovllm.offline.distributed_runner \
    --model $MODEL \
    --input-jsonl $WORKLOAD_JSONL \
    --output-dir $RUN_DIR \
    --gpus 0 \
    --planner length_bucket_token_budget \
    --shard-policy token_cost_greedy \
    --batch-size 8 --max-batch-tokens 4096 \
    --enforce-eager

# Run 2-GPU DP (only if NUM_GPUS >= 2)
if [ "$NUM_GPUS" -ge 2 ]; then
    RUN_DIR=$OUTPUT/run_2gpu
    echo "=== 2 GPU data parallel ==="
    $PYTHON -m nanovllm.offline.distributed_runner \
        --model $MODEL \
        --input-jsonl $WORKLOAD_JSONL \
        --output-dir $RUN_DIR \
        --gpus $GPUS \
        --planner prefix_then_length_bucket_token_budget \
        --shard-policy token_cost_greedy \
        --batch-size 8 --max-batch-tokens 4096 \
        --enforce-eager
fi

echo "All runs complete!"

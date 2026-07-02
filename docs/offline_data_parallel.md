# Offline Data Parallel Design

## Overview

For enterprise workloads with thousands of requests, a single GPU becomes
the bottleneck. Data Parallel (DP) distributes the workload across multiple
GPUs, with each GPU running a complete model copy on a different data shard.

## Architecture

```
                    distributed_runner.py
                         │
              ┌──────────┼──────────┐
              │          │          │
         worker_0   worker_1   worker_2
         (GPU 0)    (GPU 1)    (GPU 2)
              │          │          │
         shard_0     shard_1    shard_2
              │          │          │
         results_0   results_1  results_2
              │          │          │
              └──────────┼──────────┘
                         │
                   result_merger
                         │
                  final_output.jsonl
```

## Key Design Decisions

### 1. Subprocess-based Workers
Each worker runs as a separate Python process with `CUDA_VISIBLE_DEVICES`
set to its assigned GPU. This ensures:
- Clean GPU memory isolation (no shared CUDA context)
- Independent failure (one worker crash doesn't affect others)
- Easy monitoring via per-worker log files

### 2. Shard Policies
- **count_even**: Simple round-robin. Fast but ignores request cost.
- **token_cost_greedy**: Assigns to least-loaded worker. Better balance.
- **prefix_locality_greedy**: Keeps same-prefix requests together for
  prefix cache benefits. Best for workloads with shared templates.

### 3. Checkpoint and Resume
Each worker writes a `checkpoint_worker_N.json` after completing its shard.
On resume (`--resume`), completed IDs are skipped and only remaining
requests are processed. This handles:
- Server crashes during long runs
- GPU OOM on specific requests
- Network interruptions

### 4. Result Merging
`result_merger.py` validates:
- No duplicate IDs across workers
- No missing IDs (all expected requests completed)
- Results sorted by original ID for deterministic output

## Limitations
- Each GPU loads a full model copy (not suitable for models that don't fit
  on a single GPU — use Tensor Parallel for those)
- No inter-worker request migration
- Single-machine only (multi-node is future work)

## Expected Scaling

| GPUs | Expected Speedup | Notes |
|------|-----------------|-------|
| 1    | 1.0x            | Baseline |
| 2    | ~1.8-1.9x       | Near-linear with balanced shards |
| 4    | ~3.5-3.8x       | Diminishing returns from overhead |

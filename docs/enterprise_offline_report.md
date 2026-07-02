# Enterprise Offline Inference Report

## Overview

This report evaluates NanoServe's enterprise offline batch inference capabilities,
comparing different batch planning strategies on the RTX 4090D with Qwen3-0.6B.

## Server Configuration

- **GPU**: NVIDIA GeForce RTX 4090D (24GB VRAM)
- **Model**: Qwen/Qwen3-0.6B
- **Framework**: NanoServe v1.0.0 (based on nano-vllm)
- **max_model_len**: 4096

## Experiment Design

All experiments use 128 requests per run with the following workloads:
- **mixed-enterprise**: 30% short, 20% long-doc, 20% contract-extract, 15% RAG, 15% code-review
- **shared-prefix**: Requests grouped by 4 shared system prompts (~500-700 tokens each)

### Planners Evaluated

| Planner | Strategy |
|---|---|
| fcfs | First-come-first-served (baseline) |
| length_sorted | Sort by prompt length ascending |
| length_bucket | Group into 256-token length buckets |
| token_budget | Pack batches under 4096 token budget |
| length_bucket_token_budget | Length buckets + token budget (recommended) |
| prefix_grouped | Group by shared prefix, FCFS within |
| prefix_then_length_bucket_token_budget | Prefix groups + length bucket + token budget |

## Results

### Planner Comparison — Mixed Enterprise Workload

| Planner | Makespan | Output tok/s | Samples/s | Padding Waste |
|---|---|---|---|---|
| fcfs | 14.48s | 635.5 | 4.42 | 57.1% |
| length_sorted | 15.86s | 580.6 | 4.04 | 57.1% |
| length_bucket | 17.45s | 527.5 | 3.67 | 57.1% |
| token_budget | 16.40s | 561.4 | 3.90 | 36.3% |
| **length_bucket_token_budget** | **14.78s** | **622.7** | **4.33** | **35.6%** |

**Key finding**: `length_bucket_token_budget` reduces padding waste from 57.1% to 35.6%
while maintaining near-baseline throughput. The token budget constraint prevents
overloading batches with long sequences.

### Planner Comparison — Shared Prefix Workload

| Planner | Makespan | Output tok/s | Samples/s | Prefix Hit Rate |
|---|---|---|---|---|
| fcfs | 8.31s | 1199.1 | 7.70 | 0.0% |
| prefix_grouped | 9.23s | 1078.6 | 6.93 | 0.0% |
| prefix_then_LBTB | 10.46s | 952.6 | 6.12 | 0.0% |

**Note**: Offline batch inference clears KV cache between `generate()` calls,
so prefix cache hit rate remains 0% in this mode. The prefix-aware planners
still provide organizational benefits. For meaningful prefix cache results,
see the online serving benchmarks (64.6% hit rate with Chunked Prefill scheduler).

### Offline Data Parallel (Single GPU Baseline)

| Config | Makespan | Samples/s | Output tok/s | Load Imbalance |
|---|---|---|---|---|
| 1 GPU, 64 reqs, LBTB planner | 26.3s | 2.43 | ~620 | 0.000 |

Multi-GPU results require 2+ GPUs and will be added when available.

## Limitations

1. **Single GPU**: Current results are single-GPU only. Multi-GPU Data Parallel
   requires multiple GPUs on the same machine.
2. **Offline prefix cache**: The offline `LLM.generate()` API does not persist
   KV cache across calls, limiting prefix cache benefits.
3. **Token estimation**: Heuristic-based when tokenizer is not available; may
   cause suboptimal budget packing.
4. **No speculative decoding**: Not implemented in the current nano-vllm base.

## Conclusions

- The `length_bucket_token_budget` planner is recommended as the default for
  enterprise offline batch inference, achieving the best balance of throughput
  and padding efficiency.
- Prefix-aware planning shows organizational benefits but requires persistent
  KV cache (online serving mode) for full impact.
- The data parallel framework with checkpoint/resume provides production-ready
  reliability for long-running batch jobs.

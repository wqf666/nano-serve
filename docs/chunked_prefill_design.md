# Chunked Prefill Scheduler Design

## Problem

Long prompts (512-1024 tokens) monopolize the GPU during prefill, blocking
all concurrent decode sequences. With FCFS scheduling, a single long prefill
step can delay decode ITL by 10-50ms, causing P99 ITL spikes.

## Solution

Split long prompt prefill into chunks that fit within the token budget.
This allows decode steps to run more frequently between prefill chunks,
reducing ITL jitter.

## Strategy

Each scheduling round:
1. Try decode first — schedule all running sequences (1 token each).
2. If no decode work — use full token budget for prefill.
3. Long prompts are split: only process `chunk` tokens per prefill step.
4. Partial prefill sequences stay in the waiting queue.
5. Next round: if decode exists, decode runs; otherwise prefill continues.

## Parameters

- `max_num_batched_tokens`: total token budget per step (default: 16384)
- `min_prefill_chunk_size`: minimum tokens per prefill chunk (default: 128)
  Prevents tiny inefficient prefill steps.

## Results (mixed workload, 64 requests, rate=4, concurrency=16)

| Metric | FCFS | Decode-first | Chunked Prefill |
|--------|------|--------------|-----------------|
| TTFT P50 | 57ms | 2,317ms | 927ms |
| TTFT P95 | 85ms | 4,184ms | 2,153ms |
| ITL P50 | 4.6ms | 6.1ms | 5.1ms |
| ITL P95 | 6.2ms | 8.2ms | 7.1ms |
| E2E P95 | 2.68s | 5.69s | 3.65s |
| Throughput | 989 tok/s | 938 tok/s | 941 tok/s |

## Analysis

Chunked Prefill sits between FCFS and Decode-first on all metrics:

- **TTFT**: Much better than Decode-first (927ms vs 2.3s) because prefill
  is not starved by continuous decode. Still worse than FCFS (927ms vs 57ms)
  because decode takes priority over new prefill.

- **ITL**: Better than Decode-first (5.1ms vs 6.1ms P50) because prefill
  chunks interleave with decode steps, preventing long decode gaps.
  Slightly worse than FCFS due to decode-priority ordering.

- **Head-of-line blocking**: Partially resolved. A 1024-token prompt no
  longer blocks decode for one full step. Instead, it's split into ~6
  prefill steps of 168 tokens each, interleaved with decode steps.

## Limitations

- Decode and prefill cannot run in the SAME step (model architecture
  constraint). They alternate across steps.
- The first waiting sequence always gets prefill priority, which can
  still block shorter waiting sequences (head-of-line in waiting queue).
- No preemption of partial prefill sequences.

## Next: SLO-aware Scheduler (Phase 7)

Phase 7 will address remaining limitations by dynamically adjusting
decode/prefill priority based on SLO violation risk, rather than using
a fixed ordering.

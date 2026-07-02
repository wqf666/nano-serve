# Length-Aware Batch Planning Design

## Overview

In offline batch inference, the full set of requests is known in advance.
This allows us to **reorder and group** requests to minimize padding waste
and reduce OOM risk.

## Problem

When requests with very different prompt lengths are batched together, the
engine must pad shorter prompts to match the longest one in the batch. This
wastes GPU compute. Additionally, a single very long prompt in a batch can
cause OOM if the combined token count exceeds GPU memory.

## Solution: Length-Aware Planners

### 1. `fcfs` (baseline)
No reordering. Requests are processed in arrival order.

### 2. `length_sorted`
Sort all requests by prompt length (ascending). Short prompts are processed
first, reducing head-of-line blocking from long prompts.

### 3. `length_bucket`
Group requests into buckets of `length_bucket_size` tokens (default 256).
Requests with similar prompt lengths end up in the same bucket, minimizing
intra-batch length variance.

### 4. `token_budget`
Pack requests into batches such that the total estimated tokens (prompt +
output) per batch does not exceed `max_batch_tokens`. This prevents OOM
and ensures even GPU utilization.

### 5. `length_bucket_token_budget` (recommended)
Combine length bucketing with token budget packing. First group by length
bucket, then apply token budget within each bucket. This gives the best
of both: similar-length prompts together AND controlled total tokens.

## Token Estimation

The `TokenEstimator` class provides:
- **Exact counting** via tokenizer (preferred)
- **Heuristic estimation** via character count / avg_chars_per_token

For English text, ~4 chars per token is a reasonable heuristic.

## Metrics

Each batch records:
- `batch_id`, `num_requests`
- `sum_prompt_tokens`, `sum_estimated_tokens`, `max_prompt_tokens`

Summary adds:
- `avg_batch_prompt_tokens`, `max_batch_prompt_tokens`
- `batch_padding_waste_ratio`: `(max_prompt - avg_prompt) / max_prompt`
- `planner_name`

## Output Ordering

When `--preserve-output-order` is set, the final JSONL output is sorted by
original request ID, regardless of processing order.

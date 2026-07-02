# Prefix-Aware Batch Planning Design

## Overview

Many enterprise workloads share common prefixes across requests:
- Contract extraction: same system prompt template
- Invoice processing: same schema definition
- RAG: same retrieval instructions
- Code review: same review guidelines

When these requests are batched in arbitrary order, the prefix cache cannot
reuse blocks across requests. By **grouping requests with the same prefix**
and processing them consecutively, we maximize prefix cache hit rates.

## Problem

The KV cache prefix cache uses block-level hashing (xxhash64, 256-token blocks).
If two requests share the same first N tokens (where N >= 256), the second
request can reuse cached KV blocks from the first request's prefill.

However, if requests with different prefixes are interleaved, the cache
blocks get evicted before a matching request arrives, wasting the opportunity
for reuse.

## Solution: Prefix-Aware Planners

### 1. `prefix_grouped`
Group requests by prefix key (explicit) or prefix hash (computed from first
N tokens). Process groups in order of size (largest first). Within each group,
use FCFS ordering.

### 2. `prefix_then_length_bucket`
Group by prefix, then apply length bucket sorting within each group. This
combines prefix cache benefits with padding reduction.

### 3. `prefix_then_token_budget`
Group by prefix, then apply token budget packing within each group. This
ensures both prefix reuse and controlled memory usage per batch.

### 4. `prefix_then_length_bucket_token_budget` (recommended)
The most sophisticated planner. Groups by prefix, then applies length
bucketing and token budget packing within each group. Best overall for
enterprise workloads with diverse shared prefixes.

## Prefix Key Assignment

- If `prefix_key` is set on the request (from JSONL or workload generator),
  it's used directly for grouping.
- Otherwise, the first `prefix_hash_tokens` characters of the prompt are
  hashed with MD5 to create a group key.

## Metrics

The planner tracks:
- `prefix_key`, `prefix_hash`, `prefix_group_id` per request
- `prefix_cache_hits`, `prefix_cache_misses`, `prefix_cache_hit_rate`
- `saved_prefill_tokens` (tokens reused from cache)

## Workload Scenarios

| Scenario | prefix_key | Typical prefix tokens |
|---|---|---|
| Contract extraction | contract_extract | ~500 |
| Invoice extraction | invoice_extract | ~500 |
| Code review | code_review | ~700 |
| RAG answering | rag_answer | ~400 |

## Expected Benefits

With 64 requests and 4 prefix groups of ~16 each:
- FCFS/random order: ~0% prefix cache hit rate (groups interleaved)
- Prefix-grouped order: ~64% hit rate (same as online shared-prefix benchmark)
- Saved prefill tokens: 15,000+ tokens of computation avoided

"""Batch planners for offline inference.

Planners reorder and group requests before feeding them to the LLM engine,
optimizing for padding waste reduction and OOM avoidance.

Strategies:
  fcfs:                            First-come-first-served (no reordering)
  length_sorted:                   Sort by prompt length (ascending)
  length_bucket:                   Group into fixed-size length buckets
  token_budget:                    Pack batches under a token budget
  length_bucket_token_budget:      Bucket by length, then pack under budget
"""
from dataclasses import dataclass, field
from typing import Optional

from nanovllm.offline.token_estimator import TokenEstimator


@dataclass
class BatchInfo:
    """Metadata for a single batch."""
    batch_id: int
    request_indices: list[int]  # indices into the original request list
    num_requests: int
    sum_prompt_tokens: int
    sum_estimated_tokens: int
    max_prompt_tokens: int


@dataclass
class PlanResult:
    """Result of batch planning."""
    ordered_indices: list[int]      # reordered request indices
    batches: list[BatchInfo]        # batch metadata
    planner_name: str


# ---------------------------------------------------------------------------
# Planner implementations
# ---------------------------------------------------------------------------

def plan_fcfs(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """First-come-first-served: no reordering."""
    n = len(requests)
    batch_size = kwargs.get("batch_size", 8)
    batches = []
    for bid in range(0, n, batch_size):
        chunk = list(range(bid, min(bid + batch_size, n)))
        prompt_toks = [estimator.count_tokens(requests[i].prompt) for i in chunk]
        est_toks = [estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens)
                    for i in chunk]
        batches.append(BatchInfo(
            batch_id=bid // batch_size,
            request_indices=chunk,
            num_requests=len(chunk),
            sum_prompt_tokens=sum(prompt_toks),
            sum_estimated_tokens=sum(est_toks),
            max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
        ))
    return PlanResult(list(range(n)), batches, "fcfs")


def plan_length_sorted(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """Sort by prompt length ascending (short prompts first)."""
    n = len(requests)
    batch_size = kwargs.get("batch_size", 8)
    indices = sorted(range(n), key=lambda i: estimator.count_tokens(requests[i].prompt))
    batches = []
    for bid_start in range(0, n, batch_size):
        chunk = indices[bid_start:bid_start + batch_size]
        prompt_toks = [estimator.count_tokens(requests[i].prompt) for i in chunk]
        est_toks = [estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens)
                    for i in chunk]
        batches.append(BatchInfo(
            batch_id=bid_start // batch_size,
            request_indices=chunk,
            num_requests=len(chunk),
            sum_prompt_tokens=sum(prompt_toks),
            sum_estimated_tokens=sum(est_toks),
            max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
        ))
    return PlanResult(indices, batches, "length_sorted")


def plan_length_bucket(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """Group requests into length buckets, then sort within buckets."""
    n = len(requests)
    batch_size = kwargs.get("batch_size", 8)
    bucket_size = kwargs.get("length_bucket_size", 256)

    # Compute prompt lengths and assign buckets
    indexed = [(i, estimator.count_tokens(requests[i].prompt)) for i in range(n)]
    buckets: dict[int, list[int]] = {}
    for i, plen in indexed:
        bucket_id = plen // bucket_size
        buckets.setdefault(bucket_id, []).append(i)

    # Flatten buckets in order (small buckets first)
    indices = []
    for bucket_id in sorted(buckets.keys()):
        indices.extend(buckets[bucket_id])

    # Build batch info
    batches = []
    for bid_start in range(0, n, batch_size):
        chunk = indices[bid_start:bid_start + batch_size]
        prompt_toks = [estimator.count_tokens(requests[i].prompt) for i in chunk]
        est_toks = [estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens)
                    for i in chunk]
        batches.append(BatchInfo(
            batch_id=bid_start // batch_size,
            request_indices=chunk,
            num_requests=len(chunk),
            sum_prompt_tokens=sum(prompt_toks),
            sum_estimated_tokens=sum(est_toks),
            max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
        ))
    return PlanResult(indices, batches, "length_bucket")


def plan_token_budget(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """Pack requests into batches under a total token budget."""
    n = len(requests)
    max_batch_tokens = kwargs.get("max_batch_tokens", 4096)

    # Compute costs
    costs = [(i, estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens))
             for i in range(n)]
    # Sort by cost ascending (pack small items first for better utilization)
    costs.sort(key=lambda x: x[1])

    indices = []
    batches = []
    current_batch = []
    current_tokens = 0
    bid = 0

    for i, cost in costs:
        if current_batch and current_tokens + cost > max_batch_tokens:
            # Flush current batch
            prompt_toks = [estimator.count_tokens(requests[j].prompt) for j in current_batch]
            est_toks = [estimator.estimate_total_cost(requests[j].prompt, requests[j].max_tokens)
                        for j in current_batch]
            batches.append(BatchInfo(
                batch_id=bid,
                request_indices=list(current_batch),
                num_requests=len(current_batch),
                sum_prompt_tokens=sum(prompt_toks),
                sum_estimated_tokens=sum(est_toks),
                max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
            ))
            indices.extend(current_batch)
            bid += 1
            current_batch = []
            current_tokens = 0

        current_batch.append(i)
        current_tokens += cost

    # Flush remaining
    if current_batch:
        prompt_toks = [estimator.count_tokens(requests[j].prompt) for j in current_batch]
        est_toks = [estimator.estimate_total_cost(requests[j].prompt, requests[j].max_tokens)
                    for j in current_batch]
        batches.append(BatchInfo(
            batch_id=bid,
            request_indices=list(current_batch),
            num_requests=len(current_batch),
            sum_prompt_tokens=sum(prompt_toks),
            sum_estimated_tokens=sum(est_toks),
            max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
        ))
        indices.extend(current_batch)

    return PlanResult(indices, batches, "token_budget")


def plan_length_bucket_token_budget(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """Bucket by length, then apply token budget within each bucket."""
    n = len(requests)
    max_batch_tokens = kwargs.get("max_batch_tokens", 4096)
    bucket_size = kwargs.get("length_bucket_size", 256)

    # Assign buckets
    indexed = [(i, estimator.count_tokens(requests[i].prompt)) for i in range(n)]
    buckets: dict[int, list[tuple[int, int]]] = {}
    for i, plen in indexed:
        bucket_id = plen // bucket_size
        buckets.setdefault(bucket_id, []).append((i, plen))

    indices = []
    batches = []
    bid = 0

    for bucket_id in sorted(buckets.keys()):
        bucket_items = buckets[bucket_id]
        # Sort within bucket by length
        bucket_items.sort(key=lambda x: x[1])

        current_batch = []
        current_tokens = 0

        for i, plen in bucket_items:
            cost = estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens)
            if current_batch and current_tokens + cost > max_batch_tokens:
                # Flush
                prompt_toks = [estimator.count_tokens(requests[j].prompt) for j in current_batch]
                est_toks = [estimator.estimate_total_cost(requests[j].prompt, requests[j].max_tokens)
                            for j in current_batch]
                batches.append(BatchInfo(
                    batch_id=bid,
                    request_indices=list(current_batch),
                    num_requests=len(current_batch),
                    sum_prompt_tokens=sum(prompt_toks),
                    sum_estimated_tokens=sum(est_toks),
                    max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
                ))
                indices.extend(current_batch)
                bid += 1
                current_batch = []
                current_tokens = 0

            current_batch.append(i)
            current_tokens += cost

        # Flush remaining in this bucket
        if current_batch:
            prompt_toks = [estimator.count_tokens(requests[j].prompt) for j in current_batch]
            est_toks = [estimator.estimate_total_cost(requests[j].prompt, requests[j].max_tokens)
                        for j in current_batch]
            batches.append(BatchInfo(
                batch_id=bid,
                request_indices=list(current_batch),
                num_requests=len(current_batch),
                sum_prompt_tokens=sum(prompt_toks),
                sum_estimated_tokens=sum(est_toks),
                max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
            ))
            indices.extend(current_batch)
            bid += 1

    return PlanResult(indices, batches, "length_bucket_token_budget")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


from nanovllm.offline.prefix_planner import (
    plan_prefix_grouped,
    plan_prefix_then_length_bucket,
    plan_prefix_then_token_budget,
    plan_prefix_then_length_bucket_token_budget,
)

PLANNER_REGISTRY = {
    "fcfs": plan_fcfs,
    "length_sorted": plan_length_sorted,
    "length_bucket": plan_length_bucket,
    "token_budget": plan_token_budget,
    "length_bucket_token_budget": plan_length_bucket_token_budget,
    "prefix_grouped": plan_prefix_grouped,
    "prefix_then_length_bucket": plan_prefix_then_length_bucket,
    "prefix_then_token_budget": plan_prefix_then_token_budget,
    "prefix_then_length_bucket_token_budget": plan_prefix_then_length_bucket_token_budget,
}


def create_planner(name: str):
    """Get a planner function by name."""
    if name not in PLANNER_REGISTRY:
        raise ValueError(f"Unknown planner: {name}. Available: {list(PLANNER_REGISTRY.keys())}")
    return PLANNER_REGISTRY[name]

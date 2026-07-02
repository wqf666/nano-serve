"""Prefix-aware batch planners for offline inference.

Groups requests by shared prefix (system prompt, schema, RAG template)
to maximize prefix cache hit rates. Within each prefix group, applies
length-aware or token-budget planning.

Strategies:
  prefix_grouped:                            Group by prefix, FCFS within group
  prefix_then_length_bucket:                 Group by prefix, length bucket within
  prefix_then_token_budget:                  Group by prefix, token budget within
  prefix_then_length_bucket_token_budget:    Group by prefix, best planner within
"""
import hashlib
from collections import defaultdict

from nanovllm.offline.batch_planner import (
    BatchInfo, PlanResult, plan_fcfs, plan_length_bucket,
    plan_token_budget, plan_length_bucket_token_budget,
)
from nanovllm.offline.token_estimator import TokenEstimator


def _compute_prefix_hash(prompt: str | list[int], prefix_tokens: int, estimator: TokenEstimator) -> str:
    """Compute a hash of the first prefix_tokens tokens of the prompt."""
    if isinstance(prompt, list):
        tokens = prompt[:prefix_tokens]
        return hashlib.md5(str(tokens).encode()).hexdigest()[:12]
    # For text prompts, hash first N characters (approx prefix_tokens * 4 chars)
    char_limit = prefix_tokens * 4
    prefix_text = prompt[:char_limit]
    return hashlib.md5(prefix_text.encode()).hexdigest()[:12]


def _group_by_prefix(
    requests: list,
    estimator: TokenEstimator,
    prefix_hash_tokens: int = 512,
) -> dict[str, list[int]]:
    """Group request indices by prefix key or computed prefix hash."""
    groups = defaultdict(list)
    for i, req in enumerate(requests):
        if req.prefix_key:
            key = req.prefix_key
        else:
            key = _compute_prefix_hash(req.prompt, prefix_hash_tokens, estimator)
        groups[key].append(i)
    return dict(groups)


def plan_prefix_grouped(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """Group by prefix, FCFS within each group."""
    prefix_hash_tokens = kwargs.get("prefix_hash_tokens", 512)
    batch_size = kwargs.get("batch_size", 8)
    groups = _group_by_prefix(requests, estimator, prefix_hash_tokens)

    indices = []
    batches = []
    bid = 0

    # Sort groups by size (larger groups first for better cache reuse)
    for prefix_key in sorted(groups.keys(), key=lambda k: -len(groups[k])):
        group_indices = groups[prefix_key]
        for start in range(0, len(group_indices), batch_size):
            chunk = group_indices[start:start + batch_size]
            prompt_toks = [estimator.count_tokens(requests[i].prompt) for i in chunk]
            est_toks = [estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens)
                        for i in chunk]
            batches.append(BatchInfo(
                batch_id=bid,
                request_indices=chunk,
                num_requests=len(chunk),
                sum_prompt_tokens=sum(prompt_toks),
                sum_estimated_tokens=sum(est_toks),
                max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
            ))
            indices.extend(chunk)
            bid += 1

    return PlanResult(indices, batches, "prefix_grouped")


def plan_prefix_then_length_bucket(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """Group by prefix, then apply length bucket within each group."""
    prefix_hash_tokens = kwargs.get("prefix_hash_tokens", 512)
    groups = _group_by_prefix(requests, estimator, prefix_hash_tokens)

    all_indices = []
    all_batches = []
    bid = 0

    for prefix_key in sorted(groups.keys(), key=lambda k: -len(groups[k])):
        group_indices = groups[prefix_key]
        group_requests = [requests[i] for i in group_indices]

        # Apply length bucket within group
        sub_result = plan_length_bucket(group_requests, estimator, **kwargs)

        # Map sub-indices back to original indices
        for batch in sub_result.batches:
            mapped_indices = [group_indices[j] for j in batch.request_indices]
            prompt_toks = [estimator.count_tokens(requests[i].prompt) for i in mapped_indices]
            est_toks = [estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens)
                        for i in mapped_indices]
            all_batches.append(BatchInfo(
                batch_id=bid,
                request_indices=mapped_indices,
                num_requests=len(mapped_indices),
                sum_prompt_tokens=sum(prompt_toks),
                sum_estimated_tokens=sum(est_toks),
                max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
            ))
            all_indices.extend(mapped_indices)
            bid += 1

    return PlanResult(all_indices, all_batches, "prefix_then_length_bucket")


def plan_prefix_then_token_budget(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """Group by prefix, then apply token budget within each group."""
    prefix_hash_tokens = kwargs.get("prefix_hash_tokens", 512)
    groups = _group_by_prefix(requests, estimator, prefix_hash_tokens)

    all_indices = []
    all_batches = []
    bid = 0

    for prefix_key in sorted(groups.keys(), key=lambda k: -len(groups[k])):
        group_indices = groups[prefix_key]
        group_requests = [requests[i] for i in group_indices]

        sub_result = plan_token_budget(group_requests, estimator, **kwargs)

        for batch in sub_result.batches:
            mapped_indices = [group_indices[j] for j in batch.request_indices]
            prompt_toks = [estimator.count_tokens(requests[i].prompt) for i in mapped_indices]
            est_toks = [estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens)
                        for i in mapped_indices]
            all_batches.append(BatchInfo(
                batch_id=bid,
                request_indices=mapped_indices,
                num_requests=len(mapped_indices),
                sum_prompt_tokens=sum(prompt_toks),
                sum_estimated_tokens=sum(est_toks),
                max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
            ))
            all_indices.extend(mapped_indices)
            bid += 1

    return PlanResult(all_indices, all_batches, "prefix_then_token_budget")


def plan_prefix_then_length_bucket_token_budget(requests: list, estimator: TokenEstimator, **kwargs) -> PlanResult:
    """Group by prefix, then apply length_bucket_token_budget within each group."""
    prefix_hash_tokens = kwargs.get("prefix_hash_tokens", 512)
    groups = _group_by_prefix(requests, estimator, prefix_hash_tokens)

    all_indices = []
    all_batches = []
    bid = 0

    for prefix_key in sorted(groups.keys(), key=lambda k: -len(groups[k])):
        group_indices = groups[prefix_key]
        group_requests = [requests[i] for i in group_indices]

        sub_result = plan_length_bucket_token_budget(group_requests, estimator, **kwargs)

        for batch in sub_result.batches:
            mapped_indices = [group_indices[j] for j in batch.request_indices]
            prompt_toks = [estimator.count_tokens(requests[i].prompt) for i in mapped_indices]
            est_toks = [estimator.estimate_total_cost(requests[i].prompt, requests[i].max_tokens)
                        for i in mapped_indices]
            all_batches.append(BatchInfo(
                batch_id=bid,
                request_indices=mapped_indices,
                num_requests=len(mapped_indices),
                sum_prompt_tokens=sum(prompt_toks),
                sum_estimated_tokens=sum(est_toks),
                max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
            ))
            all_indices.extend(mapped_indices)
            bid += 1

    return PlanResult(all_indices, all_batches, "prefix_then_length_bucket_token_budget")

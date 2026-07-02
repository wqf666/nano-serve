"""Shard planning for distributing requests across GPU workers.

Shard policies:
  count_even:             Split requests evenly by count
  token_cost_greedy:      Balance by estimated total token cost
  prefix_locality_greedy: Keep same-prefix requests on same worker
"""
import json
from collections import defaultdict

from nanovllm.offline.token_estimator import TokenEstimator


def shard_count_even(items: list[dict], num_workers: int) -> list[list[int]]:
    """Split indices evenly across workers."""
    shards = [[] for _ in range(num_workers)]
    for i in range(len(items)):
        shards[i % num_workers].append(i)
    return shards


def shard_token_cost_greedy(items: list[dict], num_workers: int,
                            estimator: TokenEstimator = None) -> list[list[int]]:
    """Assign requests to the worker with the lowest current token cost.

    This is a greedy bin-packing approach that balances total estimated
    tokens across workers.
    """
    if estimator is None:
        estimator = TokenEstimator()

    shards = [[] for _ in range(num_workers)]
    worker_costs = [0] * num_workers

    # Sort items by cost descending (largest first for better packing)
    indexed_costs = []
    for i, item in enumerate(items):
        prompt = item.get("prompt", "")
        max_tokens = item.get("max_tokens", 256)
        cost = estimator.estimate_total_cost(prompt, max_tokens)
        indexed_costs.append((i, cost))
    indexed_costs.sort(key=lambda x: -x[1])

    for i, cost in indexed_costs:
        # Assign to worker with minimum current cost
        min_worker = min(range(num_workers), key=lambda w: worker_costs[w])
        shards[min_worker].append(i)
        worker_costs[min_worker] += cost

    return shards


def shard_prefix_locality_greedy(items: list[dict], num_workers: int,
                                  estimator: TokenEstimator = None) -> list[list[int]]:
    """Keep requests with the same prefix_key on the same worker.

    First groups by prefix_key, then assigns groups to workers using
    greedy cost balancing.
    """
    if estimator is None:
        estimator = TokenEstimator()

    # Group by prefix_key
    groups = defaultdict(list)
    for i, item in enumerate(items):
        key = item.get("prefix_key", f"_unique_{i}")
        groups[key].append(i)

    # Compute group costs
    group_costs = []
    for key, indices in groups.items():
        total_cost = sum(
            estimator.estimate_total_cost(
                items[i].get("prompt", ""), items[i].get("max_tokens", 256)
            )
            for i in indices
        )
        group_costs.append((key, indices, total_cost))

    # Sort groups by cost descending
    group_costs.sort(key=lambda x: -x[2])

    # Greedy assign groups to workers
    shards = [[] for _ in range(num_workers)]
    worker_costs = [0] * num_workers

    for key, indices, cost in group_costs:
        min_worker = min(range(num_workers), key=lambda w: worker_costs[w])
        shards[min_worker].extend(indices)
        worker_costs[min_worker] += cost

    return shards


SHARD_POLICIES = {
    "count_even": shard_count_even,
    "token_cost_greedy": shard_token_cost_greedy,
    "prefix_locality_greedy": shard_prefix_locality_greedy,
}


def create_shard_policy(name: str):
    """Get a shard policy function by name."""
    if name not in SHARD_POLICIES:
        raise ValueError(f"Unknown shard policy: {name}. Available: {list(SHARD_POLICIES.keys())}")
    return SHARD_POLICIES[name]

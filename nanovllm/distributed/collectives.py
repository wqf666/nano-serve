"""
Collective Communication Operations for Tensor Model Parallelism
=================================================================
Thin wrappers around torch.distributed collectives scoped to the
tensor model parallel process group.  These are used by the TP linear
layers to synchronize activations across ranks.

Primitives
----------
- **all_reduce**: Sum partial results across TP ranks (used after RowParallelLinear).
- **all_gather**: Concatenate shards along dim 0 from each TP rank
  (used to reconstruct full hidden states after ColumnParallelLinear when needed).

All functions operate in-place where possible and accept an optional
``group`` argument that defaults to the TP group from ``parallel_state``.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from typing import Optional

import torch
import torch.distributed as dist

from nanovllm.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)


# ---------------------------------------------------------------------------
# All-reduce
# ---------------------------------------------------------------------------

def tensor_model_parallel_all_reduce(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """In-place SUM all-reduce across the TP group.

    This is the core primitive for row-parallel layers: each rank holds
    a partial sum and we reduce to obtain the full result.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to reduce.  Modified **in-place**.
    group : ProcessGroup, optional
        Override the default TP group.
    async_op : bool
        If True, returns a work handle instead of blocking.

    Returns
    -------
    torch.Tensor or Work
        The reduced tensor (same object as input) when ``async_op=False``,
        otherwise the async work handle.
    """
    if group is None:
        group = get_tensor_model_parallel_group()

    if async_op:
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor


# ---------------------------------------------------------------------------
# All-gather
# ---------------------------------------------------------------------------

def tensor_model_parallel_all_gather(
    tensor: torch.Tensor,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-gather along a specified dimension across the TP group.

    Each rank contributes ``tensor`` and receives the concatenation of
    all shards along ``dim``.

    Parameters
    ----------
    tensor : torch.Tensor
        The local shard.  Shape along ``dim`` must be the same on every rank.
    dim : int
        The dimension to concatenate along (default 0).
    group : ProcessGroup, optional
        Override the default TP group.

    Returns
    -------
    torch.Tensor
        A new tensor containing the concatenated result.
    """
    if group is None:
        group = get_tensor_model_parallel_group()

    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor.clone()

    # Build list of output tensors
    # torch.distributed.all_gather requires a list of tensors
    output_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(output_list, tensor, group=group)

    # Stack along the requested dimension
    # all_gather produces a list ordered by rank; cat along dim
    if dim == 0:
        return torch.cat(output_list, dim=0)
    else:
        # For non-zero dims, use torch.cat after permute or direct cat
        return torch.cat(output_list, dim=dim)


# ---------------------------------------------------------------------------
# Scatter / Gather helpers (for future pipeline parallel support)
# ---------------------------------------------------------------------------

def tensor_model_parallel_reduce_scatter(
    tensor: torch.Tensor,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Reduce-scatter along a dimension: each rank gets one shard of the sum.

    This is useful when you want to reduce *and* re-shard in a single
    communication step, reducing memory pressure compared to all_reduce
    followed by split.

    Parameters
    ----------
    tensor : torch.Tensor
        Full tensor on each rank (will be reduced and split).
    dim : int
        Dimension to reduce and scatter along.
    group : ProcessGroup, optional
        Override the default TP group.

    Returns
    -------
    torch.Tensor
        The local shard of the reduced result.
    """
    if group is None:
        group = get_tensor_model_parallel_group()

    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor.clone()

    # First all-reduce, then take our slice.
    # NOTE: This is a naive implementation. A true reduce-scatter kernel
    # would be more bandwidth-efficient but requires custom collectives.
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)

    # Split along dim and take our shard
    shards = reduced.chunk(world_size, dim=dim)
    rank = dist.get_rank(group)
    return shards[rank].contiguous()

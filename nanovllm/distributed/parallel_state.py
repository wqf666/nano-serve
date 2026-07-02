"""
Parallel State Management for Tensor Model Parallelism
======================================================
Manages the NCCL process groups used for tensor parallel execution.
Modeled after Megatron-LM's parallel_state module, simplified for
single-node, single-group tensor parallelism.

Usage:
    from nanovllm.distributed.parallel_state import (
        initialize_model_parallel,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tensor_model_parallel_group,
        destroy_model_parallel,
    )

    # After torchrun sets RANK / LOCAL_RANK / WORLD_SIZE:
    initialize_model_parallel(tensor_model_parallel_size=2)

    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    group = get_tensor_model_parallel_group()

    # ... run inference ...

    destroy_model_parallel()
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import datetime
from typing import Optional

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# The NCCL process group for tensor parallel communication
_TENSOR_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None

# Metadata cached at init time
_TENSOR_MODEL_PARALLEL_RANK: int = -1
_TENSOR_MODEL_PARALLEL_WORLD_SIZE: int = -1

# Track whether the global default process group was initialized by us
_GLOBAL_GROUP_INITIALIZED_BY_US: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_model_parallel(tensor_model_parallel_size: int) -> None:
    """Create the tensor model parallel process group.

    Parameters
    ----------
    tensor_model_parallel_size : int
        Number of GPUs participating in tensor parallelism.  Must evenly
        divide the total world size (set by torchrun).

    Raises
    ------
    RuntimeError
        If the distributed environment is not set up correctly or TP size
        is incompatible with world size.
    """
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_RANK
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _GLOBAL_GROUP_INITIALIZED_BY_US

    if _TENSOR_MODEL_PARALLEL_GROUP is not None:
        raise RuntimeError(
            "Tensor model parallel group is already initialized. "
            "Call destroy_model_parallel() before re-initializing."
        )

    if tensor_model_parallel_size < 1:
        raise ValueError(
            f"tensor_model_parallel_size must be >= 1, got {tensor_model_parallel_size}"
        )

    # Read torchrun environment
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if rank_env is None or world_size_env is None:
        raise RuntimeError(
            "RANK and WORLD_SIZE environment variables not found. "
            "This module must be launched via torchrun or equivalent."
        )

    rank = int(rank_env)
    world_size = int(world_size_env)

    if world_size % tensor_model_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"tensor_model_parallel_size ({tensor_model_parallel_size})."
        )

    # Initialize the global process group if not already done
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=60),
        )
        _GLOBAL_GROUP_INITIALIZED_BY_US = True

    # For the prototype we assume all ranks form a single TP group.
    # In a full system with pipeline parallel you would partition ranks
    # into multiple TP groups, one per pipeline stage.
    if tensor_model_parallel_size == world_size:
        # Single TP group spanning all ranks
        tp_group = dist.group.WORLD
    else:
        # Build explicit sub-groups (e.g. for DP + TP)
        num_dp_groups = world_size // tensor_model_parallel_size
        tp_group = None
        for dp_rank in range(num_dp_groups):
            ranks_in_group = list(range(
                dp_rank * tensor_model_parallel_size,
                (dp_rank + 1) * tensor_model_parallel_size,
            ))
            group = dist.new_group(ranks=ranks_in_group)
            if rank in ranks_in_group:
                tp_group = group

        if tp_group is None:
            raise RuntimeError(
                f"Rank {rank} was not assigned to any TP group. "
                "This should not happen — check your topology."
            )

    _TENSOR_MODEL_PARALLEL_GROUP = tp_group
    _TENSOR_MODEL_PARALLEL_RANK = dist.get_rank(tp_group)
    _TENSOR_MODEL_PARALLEL_WORLD_SIZE = dist.get_world_size(tp_group)

    print(
        f"[ParallelState] Initialized TP group: "
        f"rank={_TENSOR_MODEL_PARALLEL_RANK}/{_TENSOR_MODEL_PARALLEL_WORLD_SIZE} "
        f"(global_rank={rank}/{world_size})"
    )


def get_tensor_model_parallel_rank() -> int:
    """Return the tensor-model-parallel rank of the current process."""
    if _TENSOR_MODEL_PARALLEL_RANK < 0:
        raise RuntimeError(
            "Tensor model parallel is not initialized. "
            "Call initialize_model_parallel() first."
        )
    return _TENSOR_MODEL_PARALLEL_RANK


def get_tensor_model_parallel_world_size() -> int:
    """Return the number of ranks in the tensor-model-parallel group."""
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE < 0:
        raise RuntimeError(
            "Tensor model parallel is not initialized. "
            "Call initialize_model_parallel() first."
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_group() -> dist.ProcessGroup:
    """Return the NCCL process group for tensor model parallel communication."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError(
            "Tensor model parallel is not initialized. "
            "Call initialize_model_parallel() first."
        )
    return _TENSOR_MODEL_PARALLEL_GROUP


def is_initialized() -> bool:
    """Check whether the TP group has been initialized."""
    return _TENSOR_MODEL_PARALLEL_GROUP is not None


def destroy_model_parallel() -> None:
    """Tear down the tensor model parallel group and optionally the global group."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_RANK
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _GLOBAL_GROUP_INITIALIZED_BY_US

    # Only destroy explicit groups (not WORLD singleton)
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
        and _TENSOR_MODEL_PARALLEL_GROUP is not dist.group.WORLD
    ):
        dist.destroy_process_group(_TENSOR_MODEL_PARALLEL_GROUP)

    if _GLOBAL_GROUP_INITIALIZED_BY_US and dist.is_initialized():
        dist.destroy_process_group()

    _TENSOR_MODEL_PARALLEL_GROUP = None
    _TENSOR_MODEL_PARALLEL_RANK = -1
    _TENSOR_MODEL_PARALLEL_WORLD_SIZE = -1
    _GLOBAL_GROUP_INITIALIZED_BY_US = False

    print("[ParallelState] Model parallel destroyed.")

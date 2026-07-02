"""
Tensor Parallel Linear Layers
==============================
Drop-in replacements for ``torch.nn.Linear`` that partition weights
across the tensor-model-parallel group, enabling inference of models
larger than a single GPU's memory.

Design follows the Megatron-LP paper (Shoeybi et al., 2019):

  ColumnParallelLinear  --  Y = X @ A
    Weight A is split along the **output** (column) dimension.
    Each rank computes a shard of the output; no communication in the
    forward pass (the output is already partitioned).

  RowParallelLinear     --  Y = X @ B
    Weight B is split along the **input** (row) dimension.
    Each rank holds a shard of B and computes a partial product.
    An all_reduce is needed to sum the partial results.

Splitting Conventions
---------------------
For a weight of shape [out_features, in_features] stored as a 2-D
parameter (PyTorch convention):

  ColumnParallel: each rank stores rows [r*out_per_rank : (r+1)*out_per_rank]
  RowParallel:    each rank stores cols [r*in_per_rank  : (r+1)*in_per_rank]

Qwen3-0.6B Application
-----------------------
  - Q/K/V projections  ->  ColumnParallelLinear (split heads across ranks)
  - O (output) projection  ->  RowParallelLinear (all_reduce after)
  - MLP gate_proj, up_proj  ->  ColumnParallelLinear
  - MLP down_proj           ->  RowParallelLinear
  - Embedding, lm_head      ->  Replicated (full copy on each rank)
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from nanovllm.distributed.collectives import tensor_model_parallel_all_reduce


# ============================================================================
# ColumnParallelLinear
# ============================================================================

class ColumnParallelLinear(nn.Module):
    """Linear layer with weights split along the output (column) dimension.

    Forward:
        output = input @ W^T + bias   (W is the local shard)

    The output shape is [..., output_size_per_partition].  No communication
    occurs in the forward pass; the caller is responsible for gathering
    or consuming the partitioned output.

    Parameters
    ----------
    input_size : int
        Full (un-partitioned) input feature dimension.
    output_size : int
        Full (un-partitioned) output feature dimension.  Must be
        divisible by the TP world size.
    bias : bool
        Whether to include a bias term.
    gather_output : bool
        If True, all-gather the output across the TP group so the
        downstream consumer sees the full (un-partitioned) output.
        Use sparingly as it doubles memory for the output tensor.
    dtype : torch.dtype
        Parameter dtype.
    device : str or torch.device
        Device for parameter allocation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        gather_output: bool = False,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
    ):
        super().__init__()

        if not get_tensor_model_parallel_world_size():
            raise RuntimeError("Tensor model parallel is not initialized.")

        world_size = get_tensor_model_parallel_world_size()
        if output_size % world_size != 0:
            raise ValueError(
                f"output_size ({output_size}) must be divisible by "
                f"world_size ({world_size})."
            )

        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = output_size // world_size
        self.gather_output = gather_output

        # Local shard of the weight matrix
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                input_size,
                dtype=dtype,
                device=device,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.output_size_per_partition,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.register_parameter("bias", None)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize with scaled random values for numerical stability."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def load_full_weight(self, full_weight: torch.Tensor) -> None:
        """Load a full (un-partitioned) weight and extract this rank's shard.

        Parameters
        ----------
        full_weight : torch.Tensor
            Shape [output_size, input_size].
        """
        rank = get_tensor_model_parallel_rank()
        shard_start = rank * self.output_size_per_partition
        shard_end = shard_start + self.output_size_per_partition
        shard = full_weight[shard_start:shard_end].to(
            dtype=self.weight.dtype, device=self.weight.device
        )
        self.weight.data.copy_(shard)

    def load_full_bias(self, full_bias: torch.Tensor) -> None:
        """Load a full bias and extract this rank's shard."""
        if self.bias is None:
            raise RuntimeError("This layer was created without bias.")
        rank = get_tensor_model_parallel_rank()
        shard_start = rank * self.output_size_per_partition
        shard_end = shard_start + self.output_size_per_partition
        shard = full_bias[shard_start:shard_end].to(
            dtype=self.bias.dtype, device=self.bias.device
        )
        self.bias.data.copy_(shard)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ : torch.Tensor
            Shape [..., input_size].

        Returns
        -------
        torch.Tensor
            If gather_output=False: shape [..., output_size_per_partition]
            If gather_output=True:  shape [..., output_size]
        """
        # F.linear: output = input @ weight^T + bias
        output = F.linear(input_, self.weight, self.bias)

        if self.gather_output:
            # All-gather along the last dimension
            # Each rank has [..., output_size_per_partition]
            # We want [..., output_size]
            from nanovllm.distributed.collectives import tensor_model_parallel_all_gather

            # all_gather works on dim 0, so we need to transpose for last-dim gather
            # Reshape: flatten batch dims, gather, reshape back
            original_shape = output.shape
            flat_output = output.reshape(-1, self.output_size_per_partition)

            world_size = get_tensor_model_parallel_world_size()
            group = get_tensor_model_parallel_group()
            gather_list = [torch.empty_like(flat_output) for _ in range(world_size)]
            dist.all_gather(gather_list, flat_output, group=group)

            # Concat along the feature dim
            gathered = torch.cat(gather_list, dim=-1)

            # Restore batch dims
            batch_shape = original_shape[:-1]
            output = gathered.reshape(*batch_shape, self.output_size)

        return output

    def extra_repr(self) -> str:
        return (
            f"in={self.input_size}, "
            f"out_full={self.output_size}, "
            f"out_local={self.output_size_per_partition}, "
            f"gather={self.gather_output}"
        )


# ============================================================================
# RowParallelLinear
# ============================================================================

class RowParallelLinear(nn.Module):
    """Linear layer with weights split along the input (row) dimension.

    Forward:
        partial_output = input @ W_local^T   (input is already partitioned)
        output = all_reduce(partial_output) + bias

    The input is expected to be **already partitioned** (i.e., shape
    [..., input_size_per_partition]) from a preceding ColumnParallelLinear.

    Parameters
    ----------
    input_size : int
        Full (un-partitioned) input feature dimension.  Must be
        divisible by the TP world size.
    output_size : int
        Full output feature dimension (not split).
    bias : bool
        Whether to include a bias term.  Bias is NOT split (it's added
        after the all_reduce).
    input_is_parallel : bool
        If True (default), the input is already partitioned across ranks.
        If False, the layer will scatter-chunk the input automatically.
    dtype : torch.dtype
        Parameter dtype.
    device : str or torch.device
        Device for parameter allocation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        input_is_parallel: bool = True,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
    ):
        super().__init__()

        if not get_tensor_model_parallel_world_size():
            raise RuntimeError("Tensor model parallel is not initialized.")

        world_size = get_tensor_model_parallel_world_size()
        if input_size % world_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by "
                f"world_size ({world_size})."
            )

        self.input_size = input_size
        self.output_size = output_size
        self.input_size_per_partition = input_size // world_size
        self.input_is_parallel = input_is_parallel

        # Local shard of the weight matrix
        self.weight = nn.Parameter(
            torch.empty(
                output_size,
                self.input_size_per_partition,
                dtype=dtype,
                device=device,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(output_size, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize with scaled random values."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def load_full_weight(self, full_weight: torch.Tensor) -> None:
        """Load a full weight and extract this rank's shard.

        Parameters
        ----------
        full_weight : torch.Tensor
            Shape [output_size, input_size].
        """
        rank = get_tensor_model_parallel_rank()
        shard_start = rank * self.input_size_per_partition
        shard_end = shard_start + self.input_size_per_partition
        shard = full_weight[:, shard_start:shard_end].to(
            dtype=self.weight.dtype, device=self.weight.device
        )
        self.weight.data.copy_(shard)

    def load_full_bias(self, full_bias: torch.Tensor) -> None:
        """Load the full bias (bias is replicated, not partitioned)."""
        if self.bias is None:
            raise RuntimeError("This layer was created without bias.")
        self.bias.data.copy_(
            full_bias.to(dtype=self.bias.dtype, device=self.bias.device)
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ : torch.Tensor
            If input_is_parallel=True:  shape [..., input_size_per_partition]
            If input_is_parallel=False: shape [..., input_size]

        Returns
        -------
        torch.Tensor
            Shape [..., output_size] (full, not partitioned).
        """
        if not self.input_is_parallel:
            # Scatter the input: take our chunk along the last dimension
            rank = get_tensor_model_parallel_rank()
            shard_start = rank * self.input_size_per_partition
            shard_end = shard_start + self.input_size_per_partition
            input_ = input_[..., shard_start:shard_end].contiguous()

        # Compute partial matmul: each rank produces a partial result
        output = F.linear(input_, self.weight)  # No bias yet — add after all_reduce

        # All-reduce to sum partial results across ranks
        output = tensor_model_parallel_all_reduce(output)

        # Add bias after all_reduce (bias is replicated on all ranks)
        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return (
            f"in_full={self.input_size}, "
            f"in_local={self.input_size_per_partition}, "
            f"out={self.output_size}, "
            f"input_is_parallel={self.input_is_parallel}"
        )


# ============================================================================
# Utility: Split a full state_dict for TP loading
# ============================================================================

def split_state_dict_for_tp(
    state_dict: dict,
    rank: int,
    world_size: int,
    column_parallel_keys: list,
    row_parallel_keys: list,
) -> dict:
    """Split weight tensors in a state dict for tensor parallel loading.

    This is a convenience helper for model conversion scripts.

    Parameters
    ----------
    state_dict : dict
        The full (un-partitioned) model state dict.
    rank : int
        The TP rank to extract shards for.
    world_size : int
        Total TP world size.
    column_parallel_keys : list of str
        State dict keys that correspond to ColumnParallelLinear weights.
        These are split along dim 0 (output features).
    row_parallel_keys : list of str
        State dict keys that correspond to RowParallelLinear weights.
        These are split along dim 1 (input features).

    Returns
    -------
    dict
        A new state dict with split weights for the given rank.
    """
    tp_state_dict = {}
    for key, tensor in state_dict.items():
        if key in column_parallel_keys:
            out_per_rank = tensor.shape[0] // world_size
            start = rank * out_per_rank
            end = start + out_per_rank
            tp_state_dict[key] = tensor[start:end].contiguous()
        elif key in row_parallel_keys:
            in_per_rank = tensor.shape[1] // world_size
            start = rank * in_per_rank
            end = start + in_per_rank
            tp_state_dict[key] = tensor[:, start:end].contiguous()
        else:
            # Replicated weights (e.g., embedding, lm_head, layer norms)
            tp_state_dict[key] = tensor
    return tp_state_dict

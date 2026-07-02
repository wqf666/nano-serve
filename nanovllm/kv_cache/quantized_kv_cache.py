from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

"""
QuantizedKVCache — PROTOTYPE
=============================
Wraps the existing BlockManager KV storage to apply low-bit quantization
(INT8 / FP8) to key and value tensors.  The goal of this prototype is
**memory measurement and correctness verification**, not peak throughput.

Integration point
-----------------
BlockManager (nanovllm/engine/block_manager.py) allocates blocks of shape
[num_layers, 2, block_size, num_heads, head_dim] and hands them to
ModelRunner (nanovllm/engine/model_runner.py) which feeds them into
flash-attn.  This module intercepts the read / write path:

    write:  fp16 tensor  -->  quantize  -->  int8/fp8 block + scale
    read :  int8/fp8 block + scale  -->  dequantize  -->  fp16 tensor

Because dequantization happens on every attention call, throughput will
decrease.  That is an acceptable trade-off for this memory-first
prototype.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Quantization backends
# ---------------------------------------------------------------------------

def _check_fp8_support() -> bool:
    """Return True if the current device & PyTorch build support float8_e4m3fn."""
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    # FP8 requires SM89 (Ada Lovelace) or SM90 (Hopper)
    return major >= 9 or (major == 8 and torch.cuda.get_device_capability()[1] >= 9)


FP8_AVAILABLE = _check_fp8_support()


# ---------------------------------------------------------------------------
# INT8 per-block quantization
# ---------------------------------------------------------------------------

def int8_quantize(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize *tensor* (fp16/bf16) to int8 with a per-block max-abs scale.

    "Per-block" here means one scale factor per contiguous block along the
    last dimension (head_dim).  We reduce over all other dims so the scale
    shape is broadcastable.

    Returns
    -------
    int8_tensor : torch.int8
    scale       : torch.float16  (same broadcastable shape)
    """
    # Compute max-abs along every dim except the last (head_dim)
    # Resulting shape: (1, 1, ..., head_dim) — but we actually want a
    # *single* scale per "block" (the full tensor), which is simpler and
    # matches the design-doc description.
    amax = tensor.abs().max().clamp(min=1e-12)
    scale = (amax / 127.0).to(torch.float16)
    int8_tensor = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return int8_tensor, scale


def int8_dequantize(
    int8_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize int8 tensor back to fp16."""
    return int8_tensor.to(torch.float16) * scale


# ---------------------------------------------------------------------------
# FP8 quantization (cast)
# ---------------------------------------------------------------------------

def fp8_quantize(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, None]:
    """Cast *tensor* to float8_e4m3fn.  Falls back to fp16 if unsupported."""
    if FP8_AVAILABLE:
        return tensor.to(torch.float8_e4m3fn), None
    # Fallback: keep as fp16 (no memory saving, but still correct)
    return tensor.to(torch.float16), None


def fp8_dequantize(
    fp8_tensor: torch.Tensor,
    _scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cast float8_e4m3fn back to fp16.  If fallback was used, tensor is already fp16."""
    if fp8_tensor.dtype == torch.float8_e4m3fn:
        return fp8_tensor.to(torch.float16)
    return fp8_tensor.to(torch.float16)


# ---------------------------------------------------------------------------
# Stats tracker
# ---------------------------------------------------------------------------

@dataclass
class KVCacheStats:
    original_bytes: int = 0
    quantized_bytes: int = 0
    total_blocks_written: int = 0
    total_blocks_read: int = 0
    cumulative_abs_error: float = 0.0
    max_abs_error: float = 0.0
    num_error_samples: int = 0
    total_quant_time_s: float = 0.0
    total_dequant_time_s: float = 0.0

    @property
    def saving_ratio(self) -> float:
        if self.original_bytes == 0:
            return 0.0
        return 1.0 - (self.quantized_bytes / self.original_bytes)

    @property
    def avg_abs_error(self) -> float:
        if self.num_error_samples == 0:
            return 0.0
        return self.cumulative_abs_error / self.num_error_samples

    def summary(self) -> dict:
        return {
            "original_bytes": self.original_bytes,
            "quantized_bytes": self.quantized_bytes,
            "saving_ratio": round(self.saving_ratio, 4),
            "max_abs_error": self.max_abs_error,
            "avg_abs_error": round(self.avg_abs_error, 8),
            "total_blocks_written": self.total_blocks_written,
            "total_blocks_read": self.total_blocks_read,
            "total_quant_time_s": round(self.total_quant_time_s, 4),
            "total_dequant_time_s": round(self.total_dequant_time_s, 4),
        }


# ---------------------------------------------------------------------------
# QuantizedKVCache
# ---------------------------------------------------------------------------

class QuantizedKVCache:
    """Drop-in wrapper around raw KV block storage.

    Parameters
    ----------
    dtype : str
        One of ``"fp16"`` (passthrough), ``"int8"``, ``"fp8"``.
    track_errors : bool
        If True, compute max-abs error on every write by dequantizing
        immediately and comparing.  Adds overhead — use for validation only.
    """

    SUPPORTED_DTYPES = {"fp16", "int8", "fp8"}

    def __init__(
        self,
        dtype: str = "int8",
        track_errors: bool = False,
    ):
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Choose from {self.SUPPORTED_DTYPES}"
            )
        self.dtype = dtype
        self.track_errors = track_errors
        self.stats = KVCacheStats()

        # Internal storage: maps block_id -> (quantized_k, quantized_v, scale_k, scale_v)
        self._store: dict[int, tuple] = {}

        # Select quantize / dequantize functions
        if dtype == "int8":
            self._quantize_fn = int8_quantize
            self._dequantize_fn = int8_dequantize
        elif dtype == "fp8":
            self._quantize_fn = fp8_quantize
            self._dequantize_fn = fp8_dequantize
        else:  # fp16 passthrough
            self._quantize_fn = None
            self._dequantize_fn = None

    # ------------------------------------------------------------------
    # Write path: fp16 in → quantized storage
    # ------------------------------------------------------------------

    def write_block(
        self,
        block_id: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
    ) -> None:
        """Quantize and store a KV block.

        Parameters
        ----------
        block_id : int
            Identifier assigned by BlockManager.
        k_tensor, v_tensor : torch.Tensor
            Key / Value tensors in fp16 (or bf16).
        """
        orig_bytes = k_tensor.numel() * k_tensor.element_size() + \
                     v_tensor.numel() * v_tensor.element_size()
        self.stats.original_bytes += orig_bytes
        self.stats.total_blocks_written += 1

        if self.dtype == "fp16":
            # Passthrough — store as-is
            self._store[block_id] = (
                k_tensor.detach().clone(),
                v_tensor.detach().clone(),
                None,
                None,
            )
            self.stats.quantized_bytes += orig_bytes
            return

        # Quantize
        t0 = time.perf_counter()
        qk, sk = self._quantize_fn(k_tensor)
        qv, sv = self._quantize_fn(v_tensor)
        self.stats.total_quant_time_s += time.perf_counter() - t0

        # Storage size
        q_bytes = qk.numel() * qk.element_size() + qv.numel() * qv.element_size()
        if sk is not None:
            q_bytes += sk.numel() * sk.element_size()
        if sv is not None:
            q_bytes += sv.numel() * sv.element_size()
        self.stats.quantized_bytes += q_bytes

        # Error tracking (optional, expensive)
        if self.track_errors:
            dk = self._dequantize_fn(qk, sk)
            dv = self._dequantize_fn(qv, sv)
            for original, dequantized in [(k_tensor, dk), (v_tensor, dv)]:
                err = (original.to(torch.float16) - dequantized).abs()
                max_err = err.max().item()
                self.stats.max_abs_error = max(self.stats.max_abs_error, max_err)
                self.stats.cumulative_abs_error += err.sum().item()
                self.stats.num_error_samples += err.numel()

        self._store[block_id] = (
            qk.detach().clone(),
            qv.detach().clone(),
            sk.detach().clone() if sk is not None else None,
            sv.detach().clone() if sv is not None else None,
        )

    # ------------------------------------------------------------------
    # Read path: quantized storage → fp16 out
    # ------------------------------------------------------------------

    def read_block(
        self,
        block_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read and dequantize a KV block back to fp16.

        Returns
        -------
        k_tensor, v_tensor : torch.Tensor  (fp16)
        """
        if block_id not in self._store:
            raise KeyError(f"Block {block_id} not found in QuantizedKVCache")

        qk, qv, sk, sv = self._store[block_id]
        self.stats.total_blocks_read += 1

        if self.dtype == "fp16":
            return qk, qv

        t0 = time.perf_counter()
        k = self._dequantize_fn(qk, sk)
        v = self._dequantize_fn(qv, sv)
        self.stats.total_dequant_time_s += time.perf_counter() - t0

        return k, v

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_block(self, block_id: int) -> None:
        """Remove a block from the cache (called when BlockManager frees it)."""
        self._store.pop(block_id, None)

    def clear(self) -> None:
        """Drop all blocks and reset stats."""
        self._store.clear()
        self.stats = KVCacheStats()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._store

    def get_stats(self) -> dict:
        return self.stats.summary()

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"QuantizedKVCache(dtype={self.dtype!r}, "
            f"blocks={len(self._store)}, "
            f"saving_ratio={s.saving_ratio:.2%}, "
            f"max_abs_err={s.max_abs_error:.6f})"
        )


# ---------------------------------------------------------------------------
# Standalone correctness check
# ---------------------------------------------------------------------------

def verify_quantization_correctness(
    num_layers: int = 2,
    block_size: int = 16,
    num_heads: int = 8,
    head_dim: int = 64,
    num_blocks: int = 5,
    device: str = "cpu",
) -> dict:
    """Generate random KV blocks, quantize/dequantize, and measure error.

    Returns a dict with per-dtype stats suitable for logging / JSON export.
    """
    shape = (num_layers, 2, block_size, num_heads, head_dim)
    results = {}

    for dtype_name in ("fp16", "int8", "fp8"):
        cache = QuantizedKVCache(dtype=dtype_name, track_errors=True)
        for i in range(num_blocks):
            k = torch.randn(shape, dtype=torch.float16, device=device)
            v = torch.randn(shape, dtype=torch.float16, device=device)
            cache.write_block(i, k, v)
            _ = cache.read_block(i)
        results[dtype_name] = cache.get_stats()

    return results


if __name__ == "__main__":
    import json
    print("Running KV cache quantization correctness check ...")
    res = verify_quantization_correctness()
    print(json.dumps(res, indent=2))

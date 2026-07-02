# KV Cache Quantization Design Document

> **Status**: PROTOTYPE — memory-first, correctness-focused.  
> **Authors**: NanoServe Team  
> **Last Updated**: 2025-07

---

## 1. Problem Statement — The KV Cache Memory Bottleneck

In autoregressive LLM serving, every generated token requires attention over
all previously cached key-value (KV) pairs.  The KV cache grows linearly with
sequence length and batch size:

```
KV_memory = num_layers × 2 × seq_len × num_heads × head_dim × element_size
```

For a 7B model (32 layers, 32 heads, head_dim 128) at fp16:

| seq_len | KV per request | KV for batch-32 |
|---------|----------------|-----------------|
| 2 048   | ~1 GB          | ~32 GB          |
| 8 192   | ~4 GB          | ~128 GB         |
| 32 768  | ~16 GB         | ~512 GB         |

Long-document and mixed-enterprise workloads (the two primary NanoServe
benchmarks) routinely hit seq_len ≥ 8 k, making the KV cache the **single
largest GPU memory consumer** — often exceeding model weights.

Reducing KV cache footprint directly enables larger batches, longer
contexts, or both.

---

## 2. Quantization Strategies

### 2.1 INT8 — Per-Block Max-Abs Scale Quantization (Q1)

**Algorithm**

```
scale    = max(|tensor|) / 127
int8_t   = round(tensor / scale)    # clamp to [-128, 127]
fp16_out = int8_t × scale           # dequantize
```

Each KV block (as allocated by `BlockManager`) gets **one scalar scale
factor** computed over the entire block.  This is the simplest scheme and
avoids per-channel bookkeeping.

**Memory savings**: 2× (16-bit → 8-bit), minus a negligible scale tensor.

**Trade-offs**

- Quantization noise is proportional to the dynamic range of the block.
  Outlier activations (common in LLaMA-style models) can amplify error.
- Dequantization is a multiply + cast — cheap on GPU but adds latency on
  every attention forward pass.
- Prototype uses a *global* per-block scale.  A future version could switch
  to per-head or per-channel scales for better accuracy.

### 2.2 FP8 — e4m3fn Cast (Q2)

**Algorithm**

Simply cast fp16 → `torch.float8_e4m3fn` (4-bit exponent, 3-bit mantissa,
no infinity).  No explicit scale is needed for a basic cast, though a
per-tensor scale can improve accuracy in a future iteration.

**Hardware requirement**: SM89 (Ada Lovelace, e.g. RTX 4090) or SM90
(Hopper, e.g. H100).  On older GPUs the prototype **falls back to fp16**
(passthrough) so the code path is always safe to call.

**Memory savings**: 2× (16-bit → 8-bit).

**Trade-offs**

- FP8 e4m3fn has a smaller dynamic range (max ≈ 448) but higher precision
  near zero compared to INT8.  This is generally a better fit for
  attention logits which are roughly zero-centered.
- Native FP8 tensor-core support (Hopper `fp8 × fp8` matmul) could make
  dequantization free in a future flash-attn release.  Current flash-attn
  does **not** consume fp8 KV directly, so we still dequantize to fp16.

### 2.3 FP16 — Passthrough Baseline (Q0)

No quantization.  Used as the correctness and throughput reference.

---

## 3. Theoretical Memory Savings

| dtype | element_size | Theoretical saving vs fp16 |
|-------|-------------|---------------------------|
| fp16  | 2 bytes     | 0% (baseline)             |
| int8  | 1 byte      | 50%                       |
| fp8   | 1 byte      | 50%                       |

In practice the saving is slightly less than 50% because:

1. Scale tensors (INT8) add a small overhead (~0.1%).
2. PyTorch tensor metadata is constant per allocation.
3. Block-level alignment padding.

---

## 4. Integration Architecture

```
┌──────────────┐    fp16     ┌──────────────────┐  int8/fp8  ┌──────────┐
│ ModelRunner  │────────────▶│ QuantizedKVCache  │──────────▶│ GPU Mem  │
│ (flash-attn) │◀────────────│ quantize/dequant  │◀──────────│ (blocks) │
└──────────────┘    fp16     └──────────────────┘            └──────────┘
                                  │
                                  ▼
                            BlockManager
                           (block alloc /
                              eviction)
```

**Write path** (prefill / incremental decode):
1. ModelRunner produces fp16 K, V for the new token(s).
2. `QuantizedKVCache.write_block()` quantizes and stores.
3. If `track_errors=True`, immediately dequantize and compare.

**Read path** (every attention step):
1. `QuantizedKVCache.read_block()` dequantizes stored int8/fp8 → fp16.
2. ModelRunner passes fp16 K, V to flash-attn.

**Eviction**: `evict_block()` frees the quantized storage, mirroring
BlockManager's LRU policy.

---

## 5. Correctness Verification

For each dtype we compute:

- **max_abs_error** = max |original_fp16 − dequantized_fp16|
- **avg_abs_error** = mean |original_fp16 − dequantized_fp16|

Acceptance thresholds (prototype):

| dtype | max_abs_error | avg_abs_error |
|-------|--------------|---------------|
| int8  | < 0.1        | < 0.01        |
| fp8   | < 0.05       | < 0.005       |

These are sanity bounds on random tensors; real model activations may
exhibit larger errors due to outliers.

---

## 6. Trade-offs — Dequant Overhead vs Memory Savings

| Metric              | fp16 (baseline) | int8       | fp8        |
|---------------------|-----------------|------------|------------|
| KV memory           | 1×              | ~0.5×      | ~0.5×      |
| Write overhead      | none            | +quant     | +cast      |
| Read overhead       | none            | +dequant   | +cast      |
| Throughput impact   | —               | −10-30%*   | −5-20%*    |
| Flash-attn compat   | native          | dequant    | dequant    |

*Estimated; actual numbers come from the Q3 experiment script.

**Key insight**: This prototype is **memory-first**.  Throughput *will*
decrease because every attention call dequantizes.  In a production system,
the correct approach is to fuse dequantization into the attention kernel
(e.g., a custom flash-attn variant that reads int8 directly).  That
optimization is out of scope for this prototype.

---

## 7. Experiment Plan (Q3)

The companion script `scripts/run_kv_quant_experiments.sh` runs:

| Workload          | dtype         | Measured                             |
|-------------------|---------------|--------------------------------------|
| long-doc (8k)     | fp16, int8, fp8 | peak KV mem, tok/s, output match   |
| mixed-enterprise  | fp16, int8, fp8 | peak KV mem, tok/s, output match   |

**Output match** is the fraction of output tokens identical to the fp16
baseline (temperature=0 greedy decoding).

Results are written as JSON and visualized by
`benchmark/plot_kv_quant_results.py`.

---

## 8. Future Work

- **Per-head scales**: finer granularity for INT8, modest code change.
- **Fused dequant + attention kernel**: eliminate the dequant overhead.
- **FP8 native flash-attn**: once flash-attn supports fp8 KV natively,
  the read path becomes zero-cost.
- **Outlier-aware quantization**: clamp or channel-wise scales for
  activations with heavy-tailed distributions.
- **Multi-token block quantization**: amortize scale computation across
  multiple tokens in a single block.

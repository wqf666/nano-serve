# Tensor Parallel Design Document

## Overview

This document describes the tensor model parallelism (TP) strategy for the NanoServe inference system, targeting the Qwen3-0.6B model using the nano-vllm architecture.

TP splits a single model's weights across multiple GPUs so that each GPU holds a shard. During the forward pass, each rank computes its portion and uses collective communication (all_reduce, all_gather) to produce correct results.

## Target Hardware

- **Minimum**: 2x NVIDIA GPU with NVLink (recommended) or PCIe
- **Backend**: NCCL via `torch.distributed`
- **Launch**: `torchrun` with `--nproc_per_node=2`

---

## Weight Splitting Strategy

### Transformer Block Anatomy (Qwen3)

Each transformer block contains:

1. **Self-Attention**
   - Q/K/V projections: `hidden_size -> hidden_size` (or `hidden_size -> num_kv_heads * head_dim` for GQA)
   - O (output) projection: `hidden_size -> hidden_size`

2. **MLP (Feed-Forward)**
   - `gate_proj`: `hidden_size -> intermediate_size`
   - `up_proj`: `hidden_size -> intermediate_size`
   - `down_proj`: `intermediate_size -> hidden_size`

3. **Layer Norms**: `RMSNorm(hidden_size)` (replicated, not split)

### Splitting Rules

| Component | Layer Type | Split Dimension | Communication |
|-----------|-----------|----------------|---------------|
| Q projection | ColumnParallel | output (dim 0) | None |
| K projection | ColumnParallel | output (dim 0) | None |
| V projection | ColumnParallel | output (dim 0) | None |
| O projection | RowParallel | input (dim 1) | all_reduce after matmul |
| gate_proj (MLP) | ColumnParallel | output (dim 0) | None |
| up_proj (MLP) | ColumnParallel | output (dim 0) | None |
| down_proj (MLP) | RowParallel | input (dim 1) | all_reduce after matmul |
| RMSNorm | Replicated | N/A | None |
| Embedding | Replicated | N/A | None |
| lm_head | Replicated | N/A | None |

### Why This Splitting Pattern

The key insight is the **column-parallel then row-parallel** pairing:

```
[ColumnParallel QKV]  -->  attention  -->  [RowParallel O]
    (split output)         (local)         (split input, all_reduce)

[ColumnParallel gate/up]  -->  SiLU  -->  [RowParallel down]
    (split output)          (local)       (split input, all_reduce)
```

- **ColumnParallel** (split output dim): No communication in forward pass. Output is naturally partitioned across ranks.
- **RowParallel** (split input dim): Input arrives partitioned from the preceding ColumnParallel layer. One `all_reduce` at the end produces the full result.

This yields exactly **2 all_reduce calls per transformer block** (one for O projection, one for MLP down projection).

---

## Attention: Head Splitting

For Qwen3-0.6B:
- `num_attention_heads = 16`
- `num_kv_heads = 8` (Grouped Query Attention)
- `head_dim = hidden_size / num_attention_heads = 896 / 16 = 56` (check config)

With TP=2:
- Q heads per rank: `16 / 2 = 8` heads
- KV heads per rank: `8 / 2 = 4` heads

Each rank stores its own KV cache for its local heads only. This halves the KV cache memory requirement per GPU.

### KV Cache per Local Heads

```
Rank 0: caches KV for heads [0, 1, 2, 3]
Rank 1: caches KV for heads [4, 5, 6, 7]
```

No communication is needed for KV cache operations since each rank manages its own subset of heads.

---

## Embedding and LM Head

### Current Strategy: Replicated

The token embedding and language model head are **replicated** (full copy on every rank):

- **Embedding**: Shape `[vocab_size, hidden_size]`. Each rank does a full lookup, then the hidden states are naturally replicated.
- **lm_head**: Shape `[vocab_size, hidden_size]`. After the final layer norm, each rank has the full hidden state (from the RowParallel all_reduce) and computes full logits.

### Future: Vocab Parallel (Not Implemented)

For very large vocabularies, one could split along the vocabulary dimension:
- Each rank stores `[vocab_size / world_size, hidden_size]`
- Embedding requires an all_gather after lookup
- lm_head produces partial logits, requiring all_gather before argmax/top-k

This is **not implemented** in the current prototype because Qwen3-0.6B's vocabulary (151,936 tokens) fits comfortably on a single GPU.

---

## Communication Pattern Summary

Per transformer block (per token in decode phase):

1. **After QKV projection**: No communication (ColumnParallel output stays local)
2. **After attention computation**: No communication (each rank handles its heads)
3. **After O projection**: 1x `all_reduce` (RowParallel combines partial sums)
4. **After gate/up projection**: No communication (ColumnParallel output stays local)
5. **After SiLU activation**: No communication (element-wise, local)
6. **After down projection**: 1x `all_reduce` (RowParallel combines partial sums)

**Total: 2 all_reduce calls per transformer block.**

For Qwen3-0.6B with 28 layers: 56 all_reduce calls per forward pass.

---

## Data Flow Diagram

```
Input tokens (replicated)
    |
    v
[Embedding] (replicated) --> full hidden states
    |
    v
  +--- Transformer Block (x28) ---+
  |                               |
  |  RMSNorm (replicated)         |
  |    |                          |
  |    v                          |
  |  [ColumnParallel QKV]         |  <- split output, no comm
  |    |                          |
  |    v                          |
  |  Attention (local heads)      |  <- each rank: local KV cache
  |    |                          |
  |    v                          |
  |  [RowParallel O]              |  <- all_reduce #1
  |    |                          |
  |    v                          |
  |  Residual + RMSNorm           |
  |    |                          |
  |    v                          |
  |  [ColumnParallel gate, up]    |  <- split output, no comm
  |    |                          |
  |    v                          |
  |  SiLU * up                    |  <- element-wise, local
  |    |                          |
  |    v                          |
  |  [RowParallel down]           |  <- all_reduce #2
  |    |                          |
  |    v                          |
  |  Residual                     |
  +-------------------------------+
    |
    v
[Final RMSNorm] (replicated)
    |
    v
[lm_head] (replicated) --> full logits --> sampling
    |
    v
Output token (replicated)
```

---

## Limitations

### Current Prototype

1. **Qwen3 only**: The weight mapping is hardcoded for Qwen3's architecture (RMSNorm, SwiGLU MLP, GQA attention). Other architectures (Llama, Mistral, etc.) would need custom mapping.

2. **No pipeline parallelism**: Only tensor parallelism is supported. For very deep models or limited memory, pipeline parallelism (splitting layers across GPUs) would be needed in addition.

3. **Single-node only**: The prototype assumes all GPUs are on the same node (shared via NVLink or PCIe). Multi-node TP over InfiniBand/NCCL socket is not tested.

4. **No sequence parallelism**: LayerNorm and dropout operations are replicated. For very long sequences, sequence parallelism would reduce activation memory.

5. **No expert parallelism**: Not applicable to dense models like Qwen3-0.6B, but would be needed for MoE variants.

6. **Replicated embedding/lm_head**: The full vocabulary embedding is stored on each GPU. For large-vocab models this wastes memory.

### Performance Considerations

- **NCCL overhead**: Each all_reduce has latency (~5-20 us on NVLink). With 56 all_reduces per pass, that's ~0.3-1.1ms of pure communication overhead.
- **Memory savings**: With TP=2, each GPU holds ~50% of the model weights and ~50% of the KV cache.
- **Compute efficiency**: MatMuls are smaller on each GPU, which may reduce GPU utilization for very small batch sizes.

---

## File Structure

```
nanovllm/
  distributed/
    __init__.py                  # Package init
    parallel_state.py            # Process group management
    collectives.py               # all_reduce, all_gather wrappers
    tensor_parallel_layers.py    # ColumnParallelLinear, RowParallelLinear
  offline/
    tp_env_check.py              # T0: Environment validation
    tp_generate.py               # T2: TP-aware generation prototype
scripts/
  check_tp_env.sh               # T0 launcher script
  run_offline_tp_bench.sh        # T3: Benchmark script
docs/
  tensor_parallel_design.md     # This document
```

---

## References

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
- vLLM tensor parallelism: https://github.com/vllm-project/vllm
- Qwen3 model: https://huggingface.co/Qwen/Qwen3-0.6B

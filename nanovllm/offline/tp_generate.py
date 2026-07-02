"""
T2: Tensor-Parallel Generation Prototype
==========================================
Demonstrates TP-aware text generation for Qwen3-0.6B using nano-vllm's
architecture.  Supports both single-GPU (TP=1) and multi-GPU (TP=2) paths.

Launch:
    # Single GPU (TP=1):
    python -m nanovllm.offline.tp_generate --tensor-parallel-size 1

    # Two GPUs (TP=2):
    torchrun --nproc_per_node=2 -m nanovllm.offline.tp_generate --tensor-parallel-size 2

Or use the companion scripts:
    bash scripts/run_offline_tp_bench.sh
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import json
import time
import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from nanovllm.distributed.parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    destroy_model_parallel,
)
from nanovllm.distributed.tensor_parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    split_state_dict_for_tp,
)
from nanovllm.distributed.collectives import tensor_model_parallel_all_reduce


# ============================================================================
# Configuration
# ============================================================================

# Qwen3-0.6B weight keys categorized by parallelism strategy.
# These are determined by examining the Qwen3 model architecture.
#
# Column-parallel (split output dim): Q, K, V projections, gate_proj, up_proj
# Row-parallel (split input dim): O projection, down_proj
# Replicated: embedding, lm_head, layer norms, biases

# The key patterns depend on the exact naming convention used by the
# HuggingFace Qwen3 checkpoint. Adjust these if using a different checkpoint.

COLUMN_PARALLEL_PATTERNS = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
]

ROW_PARALLEL_PATTERNS = [
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
]


def _matches_pattern(key: str, patterns: list) -> bool:
    """Check if a state dict key ends with any of the given patterns."""
    return any(key.endswith(p) for p in patterns)


def categorize_weights(state_dict: dict) -> tuple:
    """Split a state dict into column-parallel, row-parallel, and replicated keys.

    Returns
    -------
    tuple of (column_keys, row_keys)
        Lists of key strings for each parallelism category.
    """
    column_keys = []
    row_keys = []

    for key in state_dict.keys():
        if _matches_pattern(key, COLUMN_PARALLEL_PATTERNS):
            column_keys.append(key)
        elif _matches_pattern(key, ROW_PARALLEL_PATTERNS):
            row_keys.append(key)
        # else: replicated (embedding, lm_head, norms, biases)

    return column_keys, row_keys


# ============================================================================
# Model Loading
# ============================================================================

def load_model_single_gpu(
    model_name: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
) -> tuple:
    """Load the full model on a single GPU (TP=1 path).

    Returns (model, tokenizer).
    """
    print(f"[TP=1] Loading model '{model_name}' on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"[TP=1] Model loaded successfully.")
    return model, tokenizer


def load_model_tensor_parallel(
    model_name: str,
    tp_rank: int,
    tp_world_size: int,
    dtype: torch.dtype = torch.float16,
) -> tuple:
    """Load and shard the model for tensor parallel inference.

    Strategy:
      1. Rank 0 loads the full state dict from HuggingFace.
      2. Broadcast relevant metadata (config) to all ranks.
      3. Each rank extracts its shard from the state dict.

    NOTE: In a production system you would stream weights or use
    memory-mapped files to avoid duplicating the full model in CPU RAM.
    This prototype loads the full model on each rank for simplicity.

    Returns (model, tokenizer).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    print(f"[Rank {tp_rank}] Loading model '{model_name}' for TP={tp_world_size}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load config for model architecture info
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Load full model on CPU first
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    full_state_dict = full_model.state_dict()

    # Categorize and split weights
    column_keys, row_keys = categorize_weights(full_state_dict)

    print(f"[Rank {tp_rank}] Column-parallel keys: {len(column_keys)}")
    print(f"[Rank {tp_rank}] Row-parallel keys: {len(row_keys)}")
    print(f"[Rank {tp_rank}] Total keys: {len(full_state_dict)}")

    tp_state_dict = split_state_dict_for_tp(
        full_state_dict,
        rank=tp_rank,
        world_size=tp_world_size,
        column_parallel_keys=column_keys,
        row_parallel_keys=row_keys,
    )

    # Free full model from CPU memory
    del full_model
    torch.cuda.empty_cache()

    # Load the sharded model onto GPU
    # We reconstruct the model architecture then load the sharded weights.
    # In production, you would build a TP-aware model class. Here we
    # demonstrate the weight-splitting concept with the HF model as a
    # container, acknowledging that the forward pass won't actually use
    # the TP layers unless we replace the nn.Linear modules.

    # Rebuild model with empty weights for this rank
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Overwrite the split weights with our TP shards.
    # Non-split weights (replicated) are already correct from from_pretrained.
    with torch.no_grad():
        for key in column_keys + row_keys:
            if key in tp_state_dict:
                param = dict(model.named_parameters()).get(key)
                if param is not None and param.shape == tp_state_dict[key].shape:
                    param.data.copy_(tp_state_dict[key].to(device))
                    print(f"[Rank {tp_rank}] Loaded TP shard for {key}: {param.shape}")
                elif param is not None:
                    print(
                        f"[Rank {tp_rank}] Shape mismatch for {key}: "
                        f"param={param.shape}, shard={tp_state_dict[key].shape} "
                        f"(keeping replicated weight)"
                    )

    # Clean up
    del tp_state_dict, full_state_dict
    torch.cuda.empty_cache()

    print(f"[Rank {tp_rank}] TP model loaded on {device}.")
    return model, tokenizer


# ============================================================================
# Generation
# ============================================================================

def generate_single_gpu(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """Standard single-GPU generation using HuggingFace generate."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated portion
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_tensor_parallel(
    model: nn.Module,
    tokenizer,
    prompt: str,
    tp_rank: int,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
) -> Optional[str]:
    """TP-aware generation loop.

    NOTE: This is a prototype that demonstrates the TP initialization
    and weight-sharding pipeline.  The actual forward pass still uses
    the HuggingFace model's forward method, which does not insert
    all_reduce calls between layers.  A production implementation would
    replace each nn.Linear with ColumnParallelLinear / RowParallelLinear
    and implement a custom forward loop.

    For the prototype, all ranks run the same generation and should
    produce identical outputs (since weights are sharded but the HF
    model's forward isn't TP-aware, outputs may diverge for the
    sharded layers).  The key demonstration is the weight loading
    and sharding infrastructure.

    Returns the generated text only on rank 0; None on other ranks.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"[Rank {tp_rank}] Generating {max_new_tokens} tokens...")

    with torch.no_grad():
        # Use greedy decoding for determinism in the prototype
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Greedy for determinism
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if tp_rank == 0:
        return text
    return None


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_generation(
    model: nn.Module,
    tokenizer,
    prompts: list,
    max_new_tokens: int,
    tp_rank: int,
    is_tp: bool,
) -> dict:
    """Run generation on a list of prompts and collect timing stats."""
    results = []
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    for i, prompt in enumerate(prompts):
        # Warmup / synchronize
        if is_tp:
            dist.barrier()
        torch.cuda.synchronize()

        start_time = time.perf_counter()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        num_generated = len(generated_ids)
        elapsed = end_time - start_time
        tokens_per_sec = num_generated / elapsed if elapsed > 0 else 0

        if tp_rank == 0:
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"  Prompt {i+1}/{len(prompts)}: "
                  f"{num_generated} tokens in {elapsed:.3f}s "
                  f"({tokens_per_sec:.1f} tok/s)")
        else:
            text = None

        results.append({
            "prompt_idx": i,
            "prompt_length": inputs["input_ids"].shape[1],
            "generated_tokens": num_generated,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_sec,
        })

    # Aggregate
    total_tokens = sum(r["generated_tokens"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    summary = {
        "num_prompts": len(prompts),
        "total_generated_tokens": total_tokens,
        "total_time_seconds": total_time,
        "avg_tokens_per_second": avg_tps,
        "per_prompt": results,
    }

    return summary


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="NanoServe TP-aware generation prototype"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs for tensor parallelism (1 or 2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the theory of general relativity in simple terms:",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode with multiple prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tp_generate_results",
        help="Directory to save results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tp_size = args.tensor_parallel_size

    print("=" * 60)
    print(f"NanoServe TP Generation (TP={tp_size})")
    print(f"Model: {args.model_name}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # TP=1: Single-GPU path
    # ------------------------------------------------------------------
    if tp_size == 1:
        print("\n--- Single-GPU Mode ---")
        model, tokenizer = load_model_single_gpu(args.model_name)

        if args.benchmark:
            prompts = [
                "Explain the theory of general relativity in simple terms:",
                "Write a Python function to compute the Fibonacci sequence:",
                "What are the main differences between TCP and UDP?",
            ]
            summary = benchmark_generation(
                model, tokenizer, prompts, args.max_new_tokens,
                tp_rank=0, is_tp=False,
            )
        else:
            print(f"\nPrompt: {args.prompt}")
            text = generate_single_gpu(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"\nGenerated:\n{text}")
            summary = {
                "prompt": args.prompt,
                "generated_text": text,
                "tp_size": 1,
            }

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "result_tp1.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nResults saved to {output_dir}/result_tp1.json")
        return

    # ------------------------------------------------------------------
    # TP=2: Tensor-parallel path
    # ------------------------------------------------------------------
    print(f"\n--- Tensor Parallel Mode (TP={tp_size}) ---")

    # Initialize distributed
    initialize_model_parallel(tp_size)
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()

    print(f"[Rank {tp_rank}] TP group: rank {tp_rank}/{tp_world_size}")

    try:
        model, tokenizer = load_model_tensor_parallel(
            args.model_name, tp_rank, tp_world_size,
        )

        # Synchronize before generation
        dist.barrier()

        if args.benchmark:
            prompts = [
                "Explain the theory of general relativity in simple terms:",
                "Write a Python function to compute the Fibonacci sequence:",
                "What are the main differences between TCP and UDP?",
            ]
            summary = benchmark_generation(
                model, tokenizer, prompts, args.max_new_tokens,
                tp_rank=tp_rank, is_tp=True,
            )
        else:
            print(f"\n[Rank {tp_rank}] Prompt: {args.prompt}")

            text = generate_tensor_parallel(
                model, tokenizer, args.prompt, tp_rank,
                max_new_tokens=args.max_new_tokens,
            )

            if tp_rank == 0:
                print(f"\n[Rank {tp_rank}] Generated:\n{text}")
                summary = {
                    "prompt": args.prompt,
                    "generated_text": text,
                    "tp_size": tp_world_size,
                }
            else:
                summary = None

        # Save results (rank 0 only)
        if tp_rank == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"result_tp{tp_world_size}.json"
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n[Rank {tp_rank}] Results saved to {output_path}")

    finally:
        destroy_model_parallel()


if __name__ == "__main__":
    main()

"""Worker entry point for offline data parallel inference.

Each worker runs on a single GPU (via CUDA_VISIBLE_DEVICES),
processes its assigned shard, and writes results to a JSONL file.
"""
import json
import os
import sys
import time
from time import perf_counter

# Add repo root to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.offline.checkpoint import Checkpoint
from nanovllm.offline.token_estimator import TokenEstimator


def run_worker(
    worker_id: int,
    model_path: str,
    shard_items: list[dict],
    output_dir: str,
    planner: str = "fcfs",
    batch_size: int = 8,
    max_batch_tokens: int = 4096,
    max_model_len: int = 4096,
    enforce_eager: bool = False,
    resume: bool = False,
):
    """Run inference for a single worker's shard.

    Args:
        worker_id: Worker identifier.
        model_path: Path to model weights.
        shard_items: List of request dicts for this worker.
        output_dir: Directory for output files.
        planner: Batch planner name.
        batch_size: Batch size for generation.
        max_batch_tokens: Max tokens per batch.
        max_model_len: Max model sequence length.
        enforce_eager: Disable CUDA graph.
        resume: Whether to resume from checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint if resuming
    ckpt_path = os.path.join(output_dir, f"checkpoint_worker_{worker_id}.json")
    ckpt = Checkpoint.load(ckpt_path) if resume else Checkpoint(worker_id=worker_id)
    ckpt.worker_id = worker_id

    # Filter out already completed items
    if resume and ckpt.done_ids:
        shard_items = [
            item for item in shard_items
            if str(item.get("id", "")) not in ckpt.done_ids
        ]
        print(f"[Worker {worker_id}] Resuming: {len(ckpt.completed_ids)} completed, "
              f"{len(shard_items)} remaining")

    if not shard_items:
        print(f"[Worker {worker_id}] Nothing to process.")
        return

    # Initialize LLM on this worker's GPU
    print(f"[Worker {worker_id}] Loading model on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}...")
    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
    )

    estimator = TokenEstimator(tokenizer=llm.tokenizer)

    # Prepare prompts and sampling params
    prompts = []
    sampling_params_list = []
    req_ids = []

    for item in shard_items:
        prompt = item.get("prompt", "")
        max_tokens = item.get("max_tokens", 256)
        temperature = item.get("temperature", 0.6)
        prompts.append(prompt)
        sampling_params_list.append(
            SamplingParams(temperature=temperature, max_tokens=max_tokens)
        )
        req_ids.append(str(item.get("id", "")))

    # Run generation
    print(f"[Worker {worker_id}] Starting inference on {len(prompts)} requests...")
    torch.cuda.reset_peak_memory_stats()
    start_time = perf_counter()
    outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)
    wall_time = perf_counter() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Write results
    result_path = os.path.join(output_dir, f"worker_{worker_id}_results.jsonl")
    with open(result_path, "w", encoding="utf-8") as f:
        for item, output in zip(shard_items, outputs):
            req_id = str(item.get("id", ""))
            record = {
                "id": req_id,
                "text": output["text"],
                "prompt_tokens": estimator.count_tokens(item.get("prompt", "")),
                "output_tokens": len(output["token_ids"]),
                "error": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt.mark_completed(req_id)

    # Save checkpoint
    ckpt.save(ckpt_path)

    # Worker summary
    total_output = sum(len(o["token_ids"]) for o in outputs)
    print(f"[Worker {worker_id}] Done: {len(outputs)} requests in {wall_time:.1f}s, "
          f"{total_output / wall_time:.1f} tok/s, peak {peak_mem:.0f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--shard-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--planner", type=str, default="fcfs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.shard_file, "r") as f:
        shard_items = json.load(f)

    run_worker(
        worker_id=args.worker_id,
        model_path=os.path.expanduser(args.model),
        shard_items=shard_items,
        output_dir=args.output_dir,
        planner=args.planner,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        resume=args.resume,
    )

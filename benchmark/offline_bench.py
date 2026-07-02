"""Offline batch inference benchmark for NanoServe.

Usage:
    # Quick smoke test
    python benchmark/offline_bench.py --model ~/huggingface/Qwen3-0.6B \
        --workload mixed-enterprise --num-requests 8 --batch-size 2

    # Full benchmark
    python benchmark/offline_bench.py --model ~/huggingface/Qwen3-0.6B \
        --workload mixed-enterprise --num-requests 64 --batch-size 8 \
        --planner fcfs --save-result results/offline_enterprise/o0_fcfs.json

    # From JSONL input file
    python benchmark/offline_bench.py --model ~/huggingface/Qwen3-0.6B \
        --input-jsonl datasets/offline_jobs/custom.jsonl \
        --planner fcfs --save-result results/offline_enterprise/custom.json
"""
import argparse
import json
import os
import sys
import time
from time import perf_counter

# Add repo root to path so benchmark/ and nanovllm/ are importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Ensure HF downloads work on AutoDL
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from tqdm import tqdm

from nanovllm import LLM, SamplingParams
from nanovllm.offline.request_schema import OfflineRequest
from nanovllm.offline.result_writer import ResultWriter, load_jsonl, save_jsonl
from benchmark.offline_workloads import generate_offline_workload, OfflineWorkloadItem
from benchmark.offline_metrics import compute_offline_summary


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def run_offline_bench(
    llm: LLM,
    requests: list[OfflineRequest],
    batch_size: int = 8,
    use_tqdm: bool = True,
) -> tuple[list[dict], float]:
    """Run offline batch inference and collect per-request metrics.

    Args:
        llm: Initialized LLM engine.
        requests: List of OfflineRequest objects.
        batch_size: Number of requests per batch (for progress tracking).
        use_tqdm: Show progress bar.

    Returns:
        (request_records, wall_time)
    """
    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = perf_counter()

    # Build prompts and sampling params
    prompts = []
    sampling_params_list = []
    req_ids = []

    for req in requests:
        prompts.append(req.prompt)
        sampling_params_list.append(
            SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens)
        )
        req_ids.append(req.id)

    # Run generation
    outputs = llm.generate(prompts, sampling_params_list, use_tqdm=use_tqdm)

    wall_time = perf_counter() - start_time
    peak_mem = get_peak_gpu_memory_mb()

    # Build request records
    request_records = []
    for req, output in zip(requests, outputs):
        record = {
            "id": req.id,
            "text": output["text"],
            "prompt_tokens": len(req.prompt) if isinstance(req.prompt, list) else 0,
            "output_tokens": len(output["token_ids"]),
            "error": None,
        }
        # If prompt was a string, count tokens from output
        if isinstance(req.prompt, str):
            # Approximate: prompt token count isn't directly available
            # after tokenization inside LLM.generate, so we estimate
            record["prompt_tokens"] = len(req.prompt) // 4  # rough estimate
        request_records.append(record)

    return request_records, wall_time, peak_mem


def main():
    parser = argparse.ArgumentParser(description="NanoServe Offline Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--workload", type=str, default="mixed-enterprise",
                        choices=["short-batch", "long-doc", "shared-prefix",
                                 "rag-batch", "code-batch", "mixed-enterprise"],
                        help="Workload type")
    parser.add_argument("--input-jsonl", type=str, default=None,
                        help="Path to JSONL input file (overrides --workload)")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Default max tokens (overridden by workload items)")
    parser.add_argument("--planner", type=str, default="fcfs",
                        help="Batch planner name (fcfs for O0 baseline)")
    parser.add_argument("--save-result", type=str, default=None,
                        help="Path to save result JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    # Expand model path
    model_path = os.path.expanduser(args.model)

    # Load or generate workload
    if args.input_jsonl:
        raw_items = load_jsonl(args.input_jsonl)
        requests = []
        for item in raw_items:
            req = OfflineRequest(
                id=str(item.get("id", len(requests))),
                prompt=item.get("prompt", item.get("prompt_token_ids", "")),
                max_tokens=item.get("max_tokens", args.max_tokens),
                temperature=item.get("temperature", 0.6),
                prefix_key=item.get("prefix_key"),
            )
            requests.append(req)
    else:
        workload_items = generate_offline_workload(
            args.workload, args.num_requests, seed=args.seed
        )
        requests = [
            OfflineRequest(
                id=item.id,
                prompt=item.prompt,
                max_tokens=item.max_tokens,
                prefix_key=item.prefix_key,
            )
            for item in workload_items
        ]

    print(f"NanoServe Offline Benchmark")
    print(f"  Model:       {model_path}")
    print(f"  Requests:    {len(requests)}")
    print(f"  Planner:     {args.planner}")
    print(f"  Batch size:  {args.batch_size}")
    print()

    # Initialize LLM
    llm = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
    )

    # Run benchmark
    request_records, wall_time, peak_mem = run_offline_bench(
        llm, requests, batch_size=args.batch_size, use_tqdm=not args.no_tqdm
    )

    # Get prefix cache stats from engine's metrics collector if available
    prefix_cache_stats = None
    if hasattr(llm, 'metrics_collector'):
        snapshot = llm.metrics_collector.snapshot()
        pc = snapshot.get("prefix_cache", {})
        prefix_cache_stats = {
            "hits": pc.get("hits", 0),
            "misses": pc.get("misses", 0),
            "reused_tokens": pc.get("saved_prefill_tokens", 0),
        }

    # Compute summary
    summary = compute_offline_summary(
        request_records=request_records,
        wall_time=wall_time,
        peak_gpu_memory_mb=peak_mem,
        prefix_cache_stats=prefix_cache_stats,
        planner_name=args.planner,
        batch_size=args.batch_size,
        max_batch_tokens=0,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Makespan:           {summary['job_makespan_sec']:.2f}s")
    print(f"  Successful:         {summary['num_successful']}/{summary['num_requests']}")
    print(f"  Samples/s:          {summary['samples_per_second']:.2f}")
    print(f"  Output tokens/s:    {summary['output_tokens_per_second']:.2f}")
    print(f"  Total tokens/s:     {summary['total_tokens_per_second']:.2f}")
    print(f"  Peak GPU memory:    {summary['peak_gpu_memory_mb']:.1f} MB")
    print(f"  Prefix cache rate:  {summary['prefix_cache']['hit_rate']:.1%}")
    print(f"  Saved prefill tok:  {summary['prefix_cache']['saved_prefill_tokens']}")
    print(f"{'='*60}")

    # Save results
    if args.save_result:
        save_dir = os.path.dirname(args.save_result)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Save summary
        with open(args.save_result, "w") as f:
            json.dump({
                "summary": summary,
                "requests": request_records,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.save_result}")


if __name__ == "__main__":
    main()

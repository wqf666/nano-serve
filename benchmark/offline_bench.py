"""Offline batch inference benchmark for NanoServe.

Usage:
    # Quick smoke test (O0 baseline)
    python benchmark/offline_bench.py --model ~/huggingface/Qwen3-0.6B \
        --workload mixed-enterprise --num-requests 8 --batch-size 2

    # With length-aware planner (O1)
    python benchmark/offline_bench.py --model ~/huggingface/Qwen3-0.6B \
        --workload mixed-enterprise --num-requests 256 \
        --planner length_bucket_token_budget --batch-size 16 \
        --max-batch-tokens 4096 --preserve-output-order \
        --save-result results/offline_enterprise/o1_length_budget.json

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

from nanovllm import LLM, SamplingParams
from nanovllm.offline.request_schema import OfflineRequest
from nanovllm.offline.result_writer import ResultWriter, load_jsonl, save_jsonl
from nanovllm.offline.token_estimator import TokenEstimator
from nanovllm.offline.batch_planner import create_planner
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
    planner_name: str = "fcfs",
    batch_size: int = 8,
    max_batch_tokens: int = 0,
    length_bucket_size: int = 256,
    preserve_output_order: bool = False,
    use_tqdm: bool = True,
) -> tuple[list[dict], float, dict]:
    """Run offline batch inference with planning and collect per-request metrics.

    Returns:
        (request_records, wall_time, plan_stats)
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Create token estimator using the LLM's tokenizer
    estimator = TokenEstimator(tokenizer=llm.tokenizer)

    # Apply batch planner
    planner_fn = create_planner(planner_name)
    plan_result = planner_fn(
        requests, estimator,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        length_bucket_size=length_bucket_size,
    )

    # Reorder requests according to plan
    ordered_requests = [requests[i] for i in plan_result.ordered_indices]
    original_positions = {idx: pos for pos, idx in enumerate(plan_result.ordered_indices)}

    # Build prompts and sampling params in planned order
    prompts = [req.prompt for req in ordered_requests]
    sampling_params_list = [
        SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens)
        for req in ordered_requests
    ]

    # Run generation
    start_time = perf_counter()
    outputs = llm.generate(prompts, sampling_params_list, use_tqdm=use_tqdm)
    wall_time = perf_counter() - start_time
    peak_mem = get_peak_gpu_memory_mb()

    # Build request records (in planned order first)
    planned_records = []
    for pos, (req, output) in enumerate(zip(ordered_requests, outputs)):
        prompt_tokens = estimator.count_tokens(req.prompt)
        record = {
            "id": req.id,
            "text": output["text"],
            "prompt_tokens": prompt_tokens,
            "output_tokens": len(output["token_ids"]),
            "error": None,
            "planned_position": pos,
            "original_position": original_positions.get(pos, pos),
        }
        if req.prefix_key:
            record["prefix_key"] = req.prefix_key
        planned_records.append(record)

    # Restore original order if requested
    if preserve_output_order:
        planned_records.sort(key=lambda r: r["original_position"])

    # Plan statistics
    plan_stats = {
        "planner_name": plan_result.planner_name,
        "num_batches": len(plan_result.batches),
        "batches": [],
    }
    for b in plan_result.batches:
        plan_stats["batches"].append({
            "batch_id": b.batch_id,
            "num_requests": b.num_requests,
            "sum_prompt_tokens": b.sum_prompt_tokens,
            "sum_estimated_tokens": b.sum_estimated_tokens,
            "max_prompt_tokens": b.max_prompt_tokens,
        })

    # Compute batch padding waste
    if plan_result.batches:
        avg_prompt = sum(b.sum_prompt_tokens / max(b.num_requests, 1) for b in plan_result.batches) / len(plan_result.batches)
        max_prompt = max(b.max_prompt_tokens for b in plan_result.batches)
        plan_stats["avg_batch_prompt_tokens"] = round(avg_prompt, 1)
        plan_stats["max_batch_prompt_tokens"] = max_prompt
        plan_stats["batch_padding_waste_ratio"] = round(
            (max_prompt - avg_prompt) / max(max_prompt, 1), 4
        )

    return planned_records, wall_time, plan_stats


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
                        help="Batch planner: fcfs, length_sorted, length_bucket, "
                             "token_budget, length_bucket_token_budget, "
                             "prefix_grouped, prefix_then_length_bucket_token_budget")
    parser.add_argument("--max-batch-tokens", type=int, default=4096,
                        help="Max total tokens per batch (for token_budget planners)")
    parser.add_argument("--length-bucket-size", type=int, default=256,
                        help="Token length bucket size (for bucket planners)")
    parser.add_argument("--prefix-hash-tokens", type=int, default=512,
                        help="Number of prefix tokens to hash for grouping")
    parser.add_argument("--prefix-key-field", type=str, default=None,
                        help="JSONL field name for prefix key")
    parser.add_argument("--preserve-output-order", action="store_true",
                        help="Restore original request order in output")
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
                prefix_key=item.get("prefix_key") or item.get(args.prefix_key_field) if args.prefix_key_field else item.get("prefix_key"),
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
    print(f"  Model:            {model_path}")
    print(f"  Requests:         {len(requests)}")
    print(f"  Planner:          {args.planner}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Max batch tokens: {args.max_batch_tokens}")
    print(f"  Bucket size:      {args.length_bucket_size}")
    print()

    # Initialize LLM
    llm = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
    )

    # Run benchmark
    request_records, wall_time, plan_stats = run_offline_bench(
        llm, requests,
        planner_name=args.planner,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        length_bucket_size=args.length_bucket_size,
        preserve_output_order=args.preserve_output_order,
        use_tqdm=not args.no_tqdm,
    )
    peak_mem = get_peak_gpu_memory_mb()

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
        max_batch_tokens=args.max_batch_tokens,
    )
    # Add plan stats
    summary["plan_stats"] = plan_stats

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Planner:            {args.planner}")
    print(f"  Makespan:           {summary['job_makespan_sec']:.2f}s")
    print(f"  Successful:         {summary['num_successful']}/{summary['num_requests']}")
    print(f"  Samples/s:          {summary['samples_per_second']:.2f}")
    print(f"  Output tokens/s:    {summary['output_tokens_per_second']:.2f}")
    print(f"  Total tokens/s:     {summary['total_tokens_per_second']:.2f}")
    print(f"  Peak GPU memory:    {summary['peak_gpu_memory_mb']:.1f} MB")
    print(f"  Prefix cache rate:  {summary['prefix_cache']['hit_rate']:.1%}")
    print(f"  Saved prefill tok:  {summary['prefix_cache']['saved_prefill_tokens']}")
    if "batches" in plan_stats:
        print(f"  Num batches:        {plan_stats['num_batches']}")
        print(f"  Padding waste:      {plan_stats.get('batch_padding_waste_ratio', 0):.1%}")
    print(f"{'='*60}")

    # Save results
    if args.save_result:
        save_dir = os.path.dirname(args.save_result)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        with open(args.save_result, "w") as f:
            json.dump({
                "summary": summary,
                "requests": request_records,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.save_result}")


if __name__ == "__main__":
    main()

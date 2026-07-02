"""Distributed runner for offline data parallel inference.

Launches multiple workers (one per GPU), each processing a shard
of the input data. Supports checkpointing and resume.

Usage:
    python -m nanovllm.offline.distributed_runner \
        --model ~/huggingface/Qwen3-0.6B \
        --input-jsonl datasets/offline_jobs/mixed.jsonl \
        --output-dir results/offline_dp/run_001 \
        --gpus 0,1 \
        --planner prefix_then_length_bucket_token_budget \
        --shard-policy token_cost_greedy \
        --batch-size 8 --max-batch-tokens 4096 \
        --resume
"""
import argparse
import json
import os
import subprocess
import sys
import time
from time import perf_counter

# Add repo root to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from nanovllm.offline.request_schema import OfflineRequest
from nanovllm.offline.result_writer import load_jsonl, save_jsonl
from nanovllm.offline.token_estimator import TokenEstimator
from nanovllm.offline.shard_planner import create_shard_policy
from nanovllm.offline.result_merger import merge_worker_results, write_merged_results
from nanovllm.offline.checkpoint import load_all_checkpoints, get_all_completed_ids
from benchmark.offline_workloads import generate_offline_workload
from benchmark.offline_metrics import compute_offline_summary


def main():
    parser = argparse.ArgumentParser(description="NanoServe Offline Data Parallel Runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-jsonl", type=str, default=None)
    parser.add_argument("--workload", type=str, default=None)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g. '0,1')")
    parser.add_argument("--planner", type=str, default="fcfs")
    parser.add_argument("--shard-policy", type=str, default="token_cost_greedy",
                        choices=["count_even", "token_cost_greedy", "prefix_locality_greedy"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    num_workers = len(gpu_ids)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or generate workload
    if args.input_jsonl:
        items = load_jsonl(args.input_jsonl)
    elif args.workload:
        workload_items = generate_offline_workload(args.workload, args.num_requests, seed=args.seed)
        items = [
            {
                "id": wi.id,
                "prompt": wi.prompt,
                "max_tokens": wi.max_tokens,
                "prefix_key": wi.prefix_key,
            }
            for wi in workload_items
        ]
    else:
        print("Error: must specify --input-jsonl or --workload")
        sys.exit(1)

    expected_ids = {str(item.get("id", i)) for i, item in enumerate(items)}

    # Assign IDs if missing
    for i, item in enumerate(items):
        if "id" not in item:
            item["id"] = f"req_{i:04d}"

    print(f"NanoServe Offline Data Parallel Runner")
    print(f"  Model:       {model_path}")
    print(f"  Requests:    {len(items)}")
    print(f"  Workers:     {num_workers} (GPUs: {gpu_ids})")
    print(f"  Shard:       {args.shard_policy}")
    print(f"  Planner:     {args.planner}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Resume:      {args.resume}")
    print()

    # Check existing checkpoints for resume
    if args.resume:
        checkpoints = load_all_checkpoints(args.output_dir, num_workers)
        completed = get_all_completed_ids(checkpoints)
        remaining = expected_ids - completed
        print(f"  Already completed: {len(completed)}")
        print(f"  Remaining: {len(remaining)}")
        if not remaining:
            print("All requests already completed!")
            # Just merge and exit
            merged, validation = merge_worker_results(
                args.output_dir, num_workers, expected_ids
            )
            write_merged_results(merged, os.path.join(args.output_dir, "final_output.jsonl"))
            print(f"Validation: {validation}")
            return
        print()

    # Shard the data
    estimator = TokenEstimator()
    shard_fn = create_shard_policy(args.shard_policy)
    shards = shard_fn(items, num_workers, estimator=estimator)

    # Write shard files
    for wid, shard_indices in enumerate(shards):
        shard_items = [items[i] for i in shard_indices]
        shard_path = os.path.join(args.output_dir, f"shard_worker_{wid}.json")
        with open(shard_path, "w") as f:
            json.dump(shard_items, f, ensure_ascii=False)
        print(f"  Worker {wid} (GPU {gpu_ids[wid]}): {len(shard_items)} requests")

    print()

    # Launch workers as subprocesses
    start_time = perf_counter()
    python_exe = sys.executable
    worker_procs = []

    for wid, gpu_id in enumerate(gpu_ids):
        shard_file = os.path.join(args.output_dir, f"shard_worker_{wid}.json")
        cmd = [
            python_exe,
            "-m", "nanovllm.offline.worker_entry",
            "--worker-id", str(wid),
            "--model", model_path,
            "--shard-file", shard_file,
            "--output-dir", args.output_dir,
            "--planner", args.planner,
            "--batch-size", str(args.batch_size),
            "--max-batch-tokens", str(args.max_batch_tokens),
            "--max-model-len", str(args.max_model_len),
        ]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        if args.resume:
            cmd.append("--resume")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = os.path.join(args.output_dir, f"worker_{wid}.log")

        print(f"  Launching worker {wid} on GPU {gpu_id}...")
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT
            )
        worker_procs.append((wid, proc, log_path))

    # Wait for all workers
    print("\nWaiting for workers to complete...")
    all_success = True
    for wid, proc, log_path in worker_procs:
        proc.wait()
        if proc.returncode != 0:
            print(f"  Worker {wid} FAILED (exit code {proc.returncode})")
            # Print last 10 lines of log
            try:
                with open(log_path) as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(f"    {line.rstrip()}")
            except Exception:
                pass
            all_success = False
        else:
            print(f"  Worker {wid} completed successfully")

    wall_time = perf_counter() - start_time

    # Merge results
    print(f"\nMerging results...")
    merged, validation = merge_worker_results(
        args.output_dir, num_workers, expected_ids
    )
    final_path = os.path.join(args.output_dir, "final_output.jsonl")
    write_merged_results(merged, final_path)

    print(f"  Total results: {validation['total_results']}")
    print(f"  Unique IDs:    {validation['unique_ids']}")
    print(f"  Duplicates:    {validation['duplicate_ids']}")
    print(f"  Missing:       {validation['missing_ids']}")
    print(f"  Valid:         {validation['is_valid']}")

    # Compute summary
    total_output_tokens = sum(r.get("output_tokens", 0) for r in merged if r.get("error") is None)
    total_input_tokens = sum(r.get("prompt_tokens", 0) for r in merged if r.get("error") is None)

    # Per-worker stats
    per_worker = {}
    for wid in range(num_workers):
        result_path = os.path.join(args.output_dir, f"worker_{wid}_results.jsonl")
        if os.path.exists(result_path):
            worker_results = load_jsonl(result_path)
            w_output = sum(r.get("output_tokens", 0) for r in worker_results)
            per_worker[f"worker_{wid}"] = {
                "num_requests": len(worker_results),
                "output_tokens": w_output,
            }

    summary = {
        "total_makespan_sec": round(wall_time, 3),
        "num_workers": num_workers,
        "gpu_ids": gpu_ids,
        "total_requests": len(items),
        "total_successful": validation["unique_ids"],
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_tokens_per_second": round((total_output_tokens + total_input_tokens) / wall_time, 2) if wall_time > 0 else 0,
        "output_tokens_per_second": round(total_output_tokens / wall_time, 2) if wall_time > 0 else 0,
        "samples_per_second": round(validation["unique_ids"] / wall_time, 2) if wall_time > 0 else 0,
        "shard_policy": args.shard_policy,
        "planner": args.planner,
        "per_worker": per_worker,
        "validation": validation,
        "load_imbalance_ratio": _compute_imbalance(per_worker),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DATA PARALLEL RESULTS")
    print(f"{'='*60}")
    print(f"  Makespan:           {summary['total_makespan_sec']:.1f}s")
    print(f"  Workers:            {num_workers}")
    print(f"  Successful:         {summary['total_successful']}/{summary['total_requests']}")
    print(f"  Output tokens/s:    {summary['output_tokens_per_second']:.1f}")
    print(f"  Samples/s:          {summary['samples_per_second']:.2f}")
    print(f"  Load imbalance:     {summary['load_imbalance_ratio']:.3f}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}/")


def _compute_imbalance(per_worker: dict) -> float:
    """Compute load imbalance ratio (0=perfect, 1=max imbalance)."""
    if not per_worker:
        return 0.0
    counts = [w["num_requests"] for w in per_worker.values()]
    if not counts or max(counts) == 0:
        return 0.0
    return round((max(counts) - min(counts)) / max(counts), 4)


if __name__ == "__main__":
    main()

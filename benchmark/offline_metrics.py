"""Metrics computation for offline batch inference.

Computes:
  job_makespan_sec:      Total wall-clock time for the entire batch
  samples_per_second:    Completed requests / makespan
  tokens_per_second:     Total tokens (input + output) / makespan
  input_tokens_per_sec:  Total input tokens / makespan
  output_tokens_per_sec: Total output tokens / makespan
  peak_gpu_memory_mb:    Peak GPU memory usage
  prefix_cache metrics:  Hit rate, saved tokens
  error stats:           success / error / OOM counts
"""
import time


def compute_offline_summary(
    request_records: list[dict],
    wall_time: float,
    peak_gpu_memory_mb: float = 0.0,
    prefix_cache_stats: dict | None = None,
    planner_name: str = "fcfs",
    batch_size: int = 0,
    max_batch_tokens: int = 0,
) -> dict:
    """Compute summary metrics for an offline batch run.

    Args:
        request_records: List of per-request result dicts.
        wall_time: Total wall-clock time in seconds.
        peak_gpu_memory_mb: Peak GPU memory in MB.
        prefix_cache_stats: Dict with hits, misses, reused_tokens.
        planner_name: Name of the batch planner used.
        batch_size: Batch size used.
        max_batch_tokens: Max batch tokens used.
    """
    successful = [r for r in request_records if r.get("error") is None]
    failed = [r for r in request_records if r.get("error") is not None]
    oom_count = sum(1 for r in failed if "OOM" in str(r.get("error", "")))

    total_input_tokens = sum(r.get("prompt_tokens", 0) for r in successful)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in successful)
    total_tokens = total_input_tokens + total_output_tokens

    summary = {
        "num_requests": len(request_records),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "oom_count": oom_count,
        "error_rate": round(len(failed) / max(len(request_records), 1), 4),
        "job_makespan_sec": round(wall_time, 3),
        "samples_per_second": round(len(successful) / wall_time, 2) if wall_time > 0 else 0,
        "total_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_tokens_per_second": round(total_input_tokens / wall_time, 2) if wall_time > 0 else 0,
        "output_tokens_per_second": round(total_output_tokens / wall_time, 2) if wall_time > 0 else 0,
        "total_tokens_per_second": round(total_tokens / wall_time, 2) if wall_time > 0 else 0,
        "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 1),
        "planner_name": planner_name,
        "batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
    }

    # Prefix cache stats
    if prefix_cache_stats:
        hits = prefix_cache_stats.get("hits", 0)
        misses = prefix_cache_stats.get("misses", 0)
        reused = prefix_cache_stats.get("reused_tokens", 0)
        total = hits + misses
        summary["prefix_cache"] = {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / max(total, 1), 4),
            "saved_prefill_tokens": reused,
        }
    else:
        summary["prefix_cache"] = {
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "saved_prefill_tokens": 0,
        }

    return summary

"""Result merger for multi-worker offline inference.

Combines per-worker JSONL outputs into a single final_output.jsonl,
validates completeness, and restores original ordering.
"""
import json
import os
from typing import Optional


def merge_worker_results(
    output_dir: str,
    num_workers: int,
    expected_ids: Optional[set[str]] = None,
    sort_by_id: bool = True,
) -> tuple[list[dict], dict]:
    """Merge results from multiple workers.

    Args:
        output_dir: Directory containing worker_N_results.jsonl files.
        num_workers: Number of workers.
        expected_ids: Set of expected request IDs (for validation).
        sort_by_id: Sort output by original ID.

    Returns:
        (merged_results, validation_report)
    """
    all_results = []
    seen_ids = set()
    duplicate_ids = set()
    per_worker_stats = {}

    for wid in range(num_workers):
        result_path = os.path.join(output_dir, f"worker_{wid}_results.jsonl")
        if not os.path.exists(result_path):
            per_worker_stats[wid] = {"status": "missing", "count": 0}
            continue

        worker_results = []
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    req_id = str(record.get("id", ""))
                    if req_id in seen_ids:
                        duplicate_ids.add(req_id)
                    seen_ids.add(req_id)
                    worker_results.append(record)
                    all_results.append(record)

        per_worker_stats[wid] = {
            "status": "ok",
            "count": len(worker_results),
        }

    # Validation
    missing_ids = set()
    if expected_ids:
        missing_ids = expected_ids - seen_ids

    validation = {
        "total_results": len(all_results),
        "unique_ids": len(seen_ids),
        "duplicate_ids": len(duplicate_ids),
        "missing_ids": len(missing_ids),
        "expected_total": len(expected_ids) if expected_ids else None,
        "per_worker": per_worker_stats,
        "is_valid": len(duplicate_ids) == 0 and len(missing_ids) == 0,
    }

    # Sort by ID if requested
    if sort_by_id:
        all_results.sort(key=lambda r: str(r.get("id", "")))

    return all_results, validation


def write_merged_results(results: list[dict], output_path: str):
    """Write merged results to JSONL."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

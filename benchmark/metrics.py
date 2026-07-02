"""Metrics computation for online LLM serving benchmarks.

Computes per-request and summary metrics:
  TTFT:          Time to first token
  ITL:           Inter-token latency (between consecutive output tokens)
  E2E latency:   Total request duration
  P50/P90/P95/P99 percentiles
  Throughput:    tokens/s and requests/s
  Error rate
"""
import time
from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    """Per-request timing record."""
    request_id: str
    start_time: float
    first_token_time: float | None = None
    token_timestamps: list[float] = field(default_factory=list)
    finish_time: float | None = None
    output_tokens: int = 0
    success: bool = True
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None

    @property
    def e2e(self) -> float | None:
        if self.finish_time and self.start_time:
            return self.finish_time - self.start_time
        return None

    @property
    def itls(self) -> list[float]:
        """Inter-token latencies between consecutive output tokens."""
        timestamps = self.token_timestamps
        if len(timestamps) < 2:
            return []
        return [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
        ]

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "start_time": self.start_time,
            "first_token_time": self.first_token_time,
            "finish_time": self.finish_time,
            "output_tokens": self.output_tokens,
            "ttft": round(self.ttft, 6) if self.ttft is not None else None,
            "e2e_latency": round(self.e2e, 6) if self.e2e is not None else None,
            "itls": [round(x, 6) for x in self.itls],
            "success": self.success,
            "error": self.error,
        }


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = min(int(p / 100.0 * (n - 1)), n - 1)
    return sorted_vals[idx]


def compute_summary(records: list[RequestRecord], wall_time: float,
                    target_ttft: float = 0.0, target_itl: float = 0.0) -> dict:
    """Compute benchmark summary from request records.

    If target_ttft/target_itl are 0, auto-compute from P50 * 1.2.
    """
    successful = [r for r in records if r.success]
    failed = [r for r in records if not r.success]

    ttfts = [r.ttft for r in successful if r.ttft is not None]
    all_itls: list[float] = []
    for r in successful:
        all_itls.extend(r.itls)
    e2es = [r.e2e for r in successful if r.e2e is not None]

    total_output_tokens = sum(r.output_tokens for r in successful)

    # Auto-compute SLO targets from baseline P50 if not specified
    p50_ttft = _percentile(ttfts, 50) if ttfts else 0
    p50_itl = _percentile(all_itls, 50) if all_itls else 0
    if target_ttft <= 0 and p50_ttft:
        target_ttft = p50_ttft * 1.2
    if target_itl <= 0 and p50_itl:
        target_itl = p50_itl * 1.2

    # SLO violation rates
    ttft_violations = sum(1 for t in ttfts if t > target_ttft) if target_ttft > 0 else 0
    itl_violations = sum(1 for t in all_itls if t > target_itl) if target_itl > 0 else 0
    ttft_violation_rate = round(ttft_violations / max(len(ttfts), 1), 4)
    itl_violation_rate = round(itl_violations / max(len(all_itls), 1), 4)
    # Goodput: fraction of requests meeting both SLOs
    good_requests = sum(
        1 for r in successful
        if (r.ttft is not None and r.ttft <= target_ttft)
        and all(itl <= target_itl for itl in r.itls)
    ) if target_ttft > 0 and target_itl > 0 else len(successful)
    goodput = round(good_requests / max(len(successful), 1), 4)

    return {
        "num_requests": len(records),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "error_rate": round(len(failed) / max(len(records), 1), 4),
        "wall_time_s": round(wall_time, 3),
        "ttft_p50": round(_percentile(ttfts, 50), 6) if ttfts else None,
        "ttft_p90": round(_percentile(ttfts, 90), 6) if ttfts else None,
        "ttft_p95": round(_percentile(ttfts, 95), 6) if ttfts else None,
        "ttft_p99": round(_percentile(ttfts, 99), 6) if ttfts else None,
        "itl_p50": round(_percentile(all_itls, 50), 6) if all_itls else None,
        "itl_p90": round(_percentile(all_itls, 90), 6) if all_itls else None,
        "itl_p95": round(_percentile(all_itls, 95), 6) if all_itls else None,
        "itl_p99": round(_percentile(all_itls, 99), 6) if all_itls else None,
        "e2e_p50": round(_percentile(e2es, 50), 6) if e2es else None,
        "e2e_p90": round(_percentile(e2es, 90), 6) if e2es else None,
        "e2e_p95": round(_percentile(e2es, 95), 6) if e2es else None,
        "e2e_p99": round(_percentile(e2es, 99), 6) if e2es else None,
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_s": round(total_output_tokens / wall_time, 2) if wall_time > 0 else 0,
        "throughput_requests_per_s": round(len(successful) / wall_time, 2) if wall_time > 0 else 0,
        "slo": {
            "target_ttft": round(target_ttft, 6),
            "target_itl": round(target_itl, 6),
            "ttft_violation_rate": ttft_violation_rate,
            "itl_violation_rate": itl_violation_rate,
            "goodput": goodput,
        },
    }

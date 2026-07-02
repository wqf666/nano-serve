"""Online benchmark harness for NanoServe LLM serving.

Measures TTFT, ITL, E2E latency, throughput, and error rates under
various workload patterns and concurrency levels.

Usage:
    # Quick smoke test
    python benchmark/online_bench.py --url http://127.0.0.1:8000/v1/completions \
        --num-requests 8 --request-rate 2 --max-concurrency 2 --workload short

    # Full benchmark
    python benchmark/online_bench.py --url http://127.0.0.1:8000/v1/completions \
        --num-requests 256 --request-rate 8 --max-concurrency 32 --workload mixed \
        --stream --save-result /root/autodl-tmp/nanoserve/results/online_baseline/fcfs_mixed.json
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.workloads import generate_workload
from benchmark.metrics import RequestRecord, compute_summary


async def send_one_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    semaphore: asyncio.Semaphore,
    priority: int = 1,
) -> RequestRecord:
    """Send a single completion request and record timing metrics."""
    request_id = f"bench-{time.time():.6f}-{random.randint(0,99999)}"
    record = RequestRecord(request_id=request_id, start_time=time.time())

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "priority": priority,
    }

    async with semaphore:
        try:
            if stream:
                async with client.stream(
                    "POST", url, json=payload, timeout=300
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        record.success = False
                        record.error = f"HTTP {resp.status_code}: {body.decode(errors='replace')}"
                        record.finish_time = time.time()
                        return record

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                token_text = choices[0].get("text", "")
                                now = time.time()
                                if token_text:
                                    if record.first_token_time is None:
                                        record.first_token_time = now
                                    record.token_timestamps.append(now)
                                    record.output_tokens += 1
                        except json.JSONDecodeError:
                            pass

                    record.finish_time = time.time()
            else:
                resp = await client.post(url, json=payload, timeout=300)
                if resp.status_code != 200:
                    record.success = False
                    record.error = f"HTTP {resp.status_code}: {resp.text}"
                    record.finish_time = time.time()
                    return record

                data = resp.json()
                now = time.time()
                record.first_token_time = now
                record.finish_time = now
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("text", "")
                    # Approximate token count from text
                    record.output_tokens = data.get("usage", {}).get(
                        "completion_tokens", len(text.split())
                    )
                    record.token_timestamps = [now]

        except httpx.TimeoutException:
            record.success = False
            record.error = "timeout"
            record.finish_time = time.time()
        except Exception as e:
            record.success = False
            record.error = str(e)
            record.finish_time = time.time()

    return record


async def run_benchmark(
    url: str,
    num_requests: int,
    request_rate: float,
    max_concurrency: int,
    workload_type: str,
    temperature: float,
    stream: bool,
    seed: int,
    priority_mix: tuple[int, int, int] = (0, 100, 0),  # (high%, normal%, low%)
) -> tuple[list[RequestRecord], float]:
    """Run the online benchmark and return (records, wall_time)."""
    items = generate_workload(workload_type, num_requests, seed=seed)
    semaphore = asyncio.Semaphore(max_concurrency)

    # Assign priorities based on mix
    rng_pri = random.Random(seed + 1000)
    priorities = []
    h_pct, n_pct, l_pct = priority_mix
    total_pct = h_pct + n_pct + l_pct
    if total_pct == 0:
        total_pct = 100
        n_pct = 100
    for _ in range(num_requests):
        r = rng_pri.randint(1, total_pct)
        if r <= h_pct:
            priorities.append(2)  # high
        elif r <= h_pct + n_pct:
            priorities.append(1)  # normal
        else:
            priorities.append(0)  # low

    # Compute inter-arrival delays
    if request_rate > 0:
        if workload_type == "bursty":
            # Bursty: 80% of requests arrive in 20% of the time
            total_time = num_requests / request_rate
            burst_time = total_time * 0.2
            normal_time = total_time * 0.8
            burst_count = int(num_requests * 0.8)
            normal_count = num_requests - burst_count

            delays = []
            if burst_count > 1:
                delays.extend([burst_time / burst_count] * burst_count)
            if normal_count > 1:
                delays.extend([normal_time / normal_count] * normal_count)
            random.Random(seed).shuffle(delays)
        else:
            # Poisson-like: exponential inter-arrival times
            mean_delay = 1.0 / request_rate
            rng = random.Random(seed)
            delays = [rng.expovariate(1.0 / mean_delay) for _ in range(num_requests)]
    else:
        delays = [0] * num_requests

    async with httpx.AsyncClient() as client:
        tasks = []
        wall_start = time.time()

        for i, item in enumerate(items):
            priority = priorities[i] if i < len(priorities) else 1
            task = asyncio.create_task(
                send_one_request(
                    client, url, item.prompt, item.max_tokens,
                    temperature, stream, semaphore, priority=priority,
                )
            )
            tasks.append(task)

            if i < len(delays):
                await asyncio.sleep(delays[i])

        records = await asyncio.gather(*tasks)
        wall_time = time.time() - wall_start

    return list(records), wall_time


def print_report(summary: dict, workload_type: str, stream: bool,
                 records: list = None):
    """Print a human-readable benchmark report."""
    print()
    print("=" * 60)
    print("  NanoServe Online Benchmark Results")
    print("=" * 60)
    print(f"  Workload:    {workload_type}")
    print(f"  Streaming:   {stream}")
    print(f"  Requests:    {summary['num_requests']} "
          f"(ok={summary['num_successful']}, fail={summary['num_failed']})")
    print(f"  Wall time:   {summary['wall_time_s']}s")
    print(f"  Error rate:  {summary['error_rate']:.1%}")

    # Count 429 rejections (admission control)
    if records:
        rejected = sum(1 for r in records if r.error and "429" in str(r.error))
        if rejected > 0:
            print(f"  Rejected (429): {rejected} ({rejected/max(len(records),1):.1%})")

    print()
    print("  Latency Percentiles (seconds):")
    print(f"  {'Metric':<12} {'P50':>10} {'P90':>10} {'P95':>10} {'P99':>10}")
    print(f"  {'-'*52}")
    for metric in ["ttft", "itl", "e2e"]:
        p50 = summary.get(f"{metric}_p50")
        p90 = summary.get(f"{metric}_p90")
        p95 = summary.get(f"{metric}_p95")
        p99 = summary.get(f"{metric}_p99")
        label = metric.upper()
        print(f"  {label:<12} "
              f"{p50 or 'N/A':>10} {p90 or 'N/A':>10} "
              f"{p95 or 'N/A':>10} {p99 or 'N/A':>10}")
    print()
    print("  Throughput:")
    print(f"  Tokens/s:    {summary['throughput_tokens_per_s']}")
    print(f"  Requests/s:  {summary['throughput_requests_per_s']}")
    print(f"  Total tokens: {summary['total_output_tokens']}")

    slo = summary.get("slo", {})
    if slo:
        print()
        print("  SLO Violations:")
        print(f"  Target TTFT: {slo.get('target_ttft', 'N/A')}s")
        print(f"  Target ITL:  {slo.get('target_itl', 'N/A')}s")
        print(f"  TTFT violation rate: {slo.get('ttft_violation_rate', 0):.1%}")
        print(f"  ITL violation rate:  {slo.get('itl_violation_rate', 0):.1%}")
        print(f"  Goodput:            {slo.get('goodput', 0):.1%}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="NanoServe Online Benchmark")
    parser.add_argument("--url", type=str,
                        default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=2.0,
                        help="Requests per second (0 = send all at once)")
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--workload", type=str, default="short",
                        choices=["short", "medium", "mixed",
                                 "shared-prefix", "bursty"])
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-result", type=str, default=None)
    parser.add_argument("--server-url", type=str,
                        default="http://127.0.0.1:8000",
                        help="Server base URL for /metrics endpoint")
    parser.add_argument("--target-ttft", type=float, default=0.0,
                        help="TTFT SLO target in seconds (0 = auto from P50*1.2)")
    parser.add_argument("--target-itl", type=float, default=0.0,
                        help="ITL SLO target in seconds (0 = auto from P50*1.2)")
    parser.add_argument("--priority-mix", type=str, default="0,100,0",
                        help="Priority distribution: high%%,normal%%,low%% (e.g. '30,50,20')")
    args = parser.parse_args()

    stream = not args.no_stream

    # Parse priority mix
    pmix = tuple(int(x) for x in args.priority_mix.split(","))
    assert len(pmix) == 3, "priority-mix must have 3 values: high,normal,low"

    print(f"Starting benchmark: {args.num_requests} requests, "
          f"rate={args.request_rate}/s, concurrency={args.max_concurrency}, "
          f"workload={args.workload}, stream={stream}, "
          f"priority=H{pmix[0]}%/N{pmix[1]}%/L{pmix[2]}%")

    records, wall_time = asyncio.run(run_benchmark(
        url=args.url,
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        workload_type=args.workload,
        temperature=args.temperature,
        stream=stream,
        seed=args.seed,
        priority_mix=pmix,
    ))

    summary = compute_summary(records, wall_time,
                              target_ttft=args.target_ttft,
                              target_itl=args.target_itl)
    print_report(summary, args.workload, stream, records=records)

    # Fetch server metrics snapshot after benchmark
    server_metrics = None
    try:
        import httpx as _httpx
        resp = _httpx.get(f"{args.server_url}/metrics", timeout=5)
        if resp.status_code == 200:
            server_metrics = resp.json()
            print(f"  Server metrics: "
                  f"KV blocks={server_metrics.get('kv_cache', {}).get('used_blocks', '?')}/"
                  f"{server_metrics.get('kv_cache', {}).get('total_blocks', '?')}, "
                  f"prefix hit_rate={server_metrics.get('prefix_cache', {}).get('hit_rate', '?')}")
    except Exception as e:
        print(f"  Server metrics unavailable: {e}")

    # Save results
    if args.save_result:
        os.makedirs(os.path.dirname(args.save_result), exist_ok=True)
        result = {
            "config": {
                "url": args.url,
                "num_requests": args.num_requests,
                "request_rate": args.request_rate,
                "max_concurrency": args.max_concurrency,
                "workload": args.workload,
                "stream": stream,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            "summary": summary,
            "server_metrics": server_metrics,
            "requests": [r.to_dict() for r in records],
        }
        with open(args.save_result, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.save_result}")


if __name__ == "__main__":
    main()

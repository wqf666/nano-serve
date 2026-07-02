# NanoServe

**SLO-Aware Online LLM Inference Serving System** — built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), extended with pluggable schedulers, admission control, priority scheduling, and production-grade benchmarking.

---

## Why NanoServe?

nano-vllm is an excellent educational reimplementation of vLLM's offline inference engine. NanoServe takes it further into the **online serving** domain — the real-world scenario where requests arrive asynchronously and latency SLOs matter. We added:

- **Online serving layer**: async HTTP server with streaming SSE, OpenAI-compatible `/v1/completions` API
- **Pluggable scheduler framework**: 5 scheduling strategies with a unified `BaseScheduler` ABC
- **KV cache metrics**: real-time block utilization and prefix cache hit-rate monitoring
- **Admission control**: queue-depth + KV utilization based overload protection (HTTP 429)
- **Priority scheduling**: 3-tier priority queues for enterprise SLA differentiation
- **Comprehensive benchmark harness**: async client with TTFT/ITL/E2E/SLO-violation/goodput metrics

## Architecture

```
                         ┌─────────────────────────────────┐
                         │         FastAPI Server           │
                         │   /v1/completions  /health       │
                         │   /metrics         /v1/models    │
                         └──────────┬──────────────────────┘
                                    │
                         ┌──────────▼──────────────────────┐
                         │         AsyncEngine               │
                         │   ThreadPoolExecutor + asyncio    │
                         │   Priority Queue · Admission Ctrl │
                         └──────────┬──────────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │         OnlineEngine                │
                    │   step() → per-seq token tracking   │
                    │   MetricsCollector (KV + prefix)    │
                    └───────────────┬───────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
   ┌────────▼────────┐   ┌─────────▼────────┐   ┌─────────▼────────┐
   │   Scheduler      │   │  BlockManager    │   │  ModelRunner     │
   │  (pluggable)     │   │  + Prefix Cache  │   │  (prefill|decode)│
   │  FCFS / Chunked  │   │  xxhash64 chain  │   │  Flash-Attn      │
   │  Decode-First    │   │  64.6% hit rate  │   │  Triton kernels  │
   │  SLO-Aware       │   └─────────────────┘   └─────────────────┘
   │  Priority        │
   └─────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/wqf666/nano-serve.git
cd nano-serve
pip install -e .
```

Requirements: Python 3.10+, CUDA 12.x, PyTorch 2.4+, flash-attn, triton.

### Offline Inference (original nano-vllm)

```python
from nanovllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=128))
```

### Online Serving (NanoServe)

```bash
# Start server with default scheduler (FCFS)
python -m nanovllm.serve.api_server --model Qwen/Qwen3-0.6B

# Or with Chunked Prefill scheduler (recommended)
python -m nanovllm.serve.api_server \
  --model Qwen/Qwen3-0.6B \
  --scheduler chunked_prefill \
  --max-prefill-chunk-size 512

# With admission control
python -m nanovllm.serve.api_server \
  --model Qwen/Qwen3-0.6B \
  --scheduler priority \
  --max-queue-depth 32
```

### Send Requests

```bash
# Standard completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "max_tokens": 128}'

# With priority (0=low, 1=normal, 2=high)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Urgent query", "max_tokens": 64, "priority": 2}'

# Streaming
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "max_tokens": 256, "stream": true}'
```

### Health & Metrics

```bash
# Health check (includes capacity signal for load balancers)
curl http://localhost:8000/health

# Prometheus-style metrics (KV utilization, prefix cache, request counts)
curl http://localhost:8000/metrics
```

## Schedulers

| Scheduler | CLI Name | Strategy | Best For |
|---|---|---|---|
| **FCFS** | `fcfs` | First-come-first-served, prefill-first | Baseline, steady traffic |
| **Chunked Prefill** | `chunked_prefill` | Prefill-first with chunk cap + fairness rotation | **Recommended** — best all-around |
| **Decode First** | `decode_first` | Prioritize running decode sequences | ⚠️ Causes prefill starvation |
| **SLO-Aware** | `slo_aware` | Two-level priority + auto-calibration | ⚠️ Degrades under no-mix constraint |
| **Priority** | `priority` | 3-tier (high/normal/low) + chunked prefill | Enterprise tiered SLA |

> **Key architectural insight**: nano-vllm's ModelRunner cannot mix prefill and decode within the same `step()`. This means decode-preferring strategies cause prefill starvation. Chunked Prefill with prefill-first ordering is the optimal strategy under this constraint.

## Benchmark Results

**Environment**: NVIDIA RTX 4090D (24GB) · Qwen3-0.6B · 64 requests per run

### Mixed Workload (70% short / 30% long, rate=4/s, concurrency=16)

| Scheduler | TTFT P50 | TTFT P95 | ITL P95 | Throughput |
|---|---|---|---|---|
| FCFS | **56.7ms** | **85.4ms** | 6.2ms | 989 tok/s |
| Chunked Prefill | 58.7ms | 92.5ms | **6.7ms** | 988 tok/s |
| Decode First | 2,317ms | 4,184ms | 8.2ms | 938 tok/s |
| SLO-Aware | 2,347ms | 3,466ms | 11.0ms | 1,287 tok/s |

### Bursty Workload (80% requests in 20% time, rate=8/s, concurrency=32)

| Scheduler | TTFT P95 | ITL P95 | TTFT Violation | ITL Violation | Goodput |
|---|---|---|---|---|---|
| FCFS | 92ms | 8.3ms | 20.3% | 35.0% | 1.6% |
| **Chunked Prefill** | **82ms** | **7.2ms** | **9.4%** | **25.9%** | **3.1%** |
| SLO-Aware | 830ms | 7.7ms | 98.4% | 28.2% | 1.6% |

### Prefix Cache (32 requests, ~700-token shared prefix, Chunked Prefill)

| Metric | Value |
|---|---|
| Cache Hit Rate | **64.6%** |
| Reused Tokens | 15,872 |
| Throughput | 732 tok/s |

### Priority Scheduling (Mixed, 30H/50N/20L vs 100% Normal)

| Metric | Normal Only | Priority Mix | Improvement |
|---|---|---|---|
| ITL Violation | 20.1% | **7.9%** | **−61%** |
| TTFT P95 | 2.20s | **1.96s** | −11% |
| Throughput | 930 tok/s | 940 tok/s | +1% |

## Project Structure

```
nanoserve/
├── nanovllm/
│   ├── serve/                  # NanoServe additions
│   │   ├── api_server.py       # FastAPI server + admission control
│   │   ├── async_engine.py     # OnlineEngine + AsyncEngine
│   │   └── protocol.py         # Request/Response schemas
│   ├── scheduler/              # Pluggable scheduler framework
│   │   ├── base.py             # BaseScheduler ABC + registry
│   │   ├── fcfs.py
│   │   ├── decode_first.py
│   │   ├── chunked_prefill.py
│   │   ├── slo_aware.py
│   │   └── priority.py
│   ├── metrics/
│   │   └── metrics_collector.py  # KV + prefix cache metrics
│   ├── engine/                 # Core engine (from nano-vllm)
│   ├── layers/                 # Attention, MLP, RMSNorm (from nano-vllm)
│   ├── models/                 # Model definitions (from nano-vllm)
│   ├── config.py
│   ├── llm.py
│   └── sampling_params.py
├── benchmark/                  # Benchmark harness
│   ├── online_bench.py         # Async HTTP benchmark client
│   ├── metrics.py              # TTFT/ITL/E2E/SLO/goodput
│   └── workloads.py            # Steady, bursty, shared-prefix, priority
├── docs/                       # Design documents
├── scripts/
└── pyproject.toml
```

## Inspiration & Credits

NanoServe is built on top of **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** by Xingkai Yu — a lightweight, from-scratch reimplementation of vLLM's core offline inference engine. We gratefully acknowledge this foundation and recommend reading the nano-vllm source code for understanding the internals of PagedAttention, block management, and model execution.

The online serving layer, scheduler framework, metrics system, benchmark harness, and all production-oriented features are original NanoServe contributions.

## License

MIT — same as nano-vllm.

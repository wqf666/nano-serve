# Changelog

All notable changes to NanoServe are documented in this file.

## [1.0.0] — 2026-07-02

### Added

**Online Serving Layer (Phase 1–2)**
- FastAPI-based HTTP server with OpenAI-compatible `/v1/completions` endpoint
- `OnlineEngine` subclassing `LLMEngine` with per-sequence token tracking via `prev_counts` snapshot
- `AsyncEngine` with `ThreadPoolExecutor` for non-blocking CUDA operations and per-request `asyncio.Queue` token dispatch
- Streaming SSE support for real-time token delivery
- `/v1/models`, `/health`, `/metrics` endpoints

**Pluggable Scheduler Framework (Phase 3–5)**
- `BaseScheduler` ABC with `SCHEDULER_REGISTRY` factory pattern
- **FCFS** — first-come-first-served baseline scheduler (migrated from engine core)
- **Decode-First** — prioritizes running decode sequences; discovered to cause prefill starvation under the no-mix constraint
- **Chunked Prefill** — prefill-first ordering with configurable chunk cap and fairness rotation; best overall scheduler
- **SLO-Aware** — two-level priority queue with auto-calibration; degrades under no-mix constraint
- **Priority** — 3-tier (high/normal/low) priority scheduling extending Chunked Prefill; 61% ITL violation reduction for tiered SLA

**KV Cache Metrics & Prefix Cache (Phase 6)**
- `MetricsCollector` with monkey-patch instrumentation on `BlockManager`
- Real-time KV block utilization tracking
- Block-level prefix cache with xxhash64 chained hashing (256-token blocks)
- 64.6% cache hit rate on shared-prefix workloads (~700-token prefix)

**Admission Control & Enterprise Features (Phase 7, A–B)**
- Queue-depth + KV utilization based overload protection returning HTTP 429
- `/health` endpoint with capacity signals for upstream load balancers
- Configurable `--max-queue-depth` threshold
- Priority-aware request admission

**Benchmark Harness**
- Async HTTP benchmark client supporting steady, bursty, shared-prefix, and priority-mix workloads
- Comprehensive metrics: TTFT (P50/P90/P95/P99), ITL, E2E, SLO violation rate, goodput, throughput
- Auto SLO target calibration from FCFS P50 × 1.2

### Changed

- Rebranded from nano-vllm to NanoServe
- Updated `pyproject.toml` with serving dependencies and new metadata
- nano-vllm credited as inspiration source in README

### Known Limitations

- ModelRunner cannot mix prefill and decode within the same `step()` — this is the fundamental architectural bottleneck for advanced scheduling strategies
- Prefix cache only effective for full blocks (multiples of 256 tokens)
- Priority scheduling effect diminishes under extreme bursty traffic

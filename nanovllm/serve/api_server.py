"""FastAPI server for NanoServe online LLM inference.

Endpoints:
  GET  /health             Health check
  GET  /v1/models          List loaded models
  POST /v1/completions     Text completion (supports stream=true)
"""
import argparse
import asyncio
import json
import logging
import os
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from nanovllm.sampling_params import SamplingParams
from nanovllm.serve.async_engine import AsyncEngine
from nanovllm.serve.protocol import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    StreamingChunk,
    StreamingChoice,
    UsageInfo,
)
from nanovllm.serve.request_tracker import RequestTracker
from nanovllm.serve.streaming import format_sse, sse_data_end

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nanoserve")

app = FastAPI(title="NanoServe")
engine: AsyncEngine | None = None
tracker = RequestTracker()


@app.on_event("startup")
async def startup():
    global engine
    args = app.state.args
    logger.info(f"Loading model: {args.model}")
    engine = AsyncEngine(
        args.model,
        scheduler_name=args.scheduler,
        scheduler_params={
            "max_prefill_chunk_size": args.max_prefill_chunk_size,
            "min_prefill_chunk_size": args.min_prefill_chunk_size,
        },
        max_queue_depth=args.max_queue_depth,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    await engine.start()
    logger.info(
        f"Model loaded. Scheduler: {engine.scheduler_name}. "
        f"Server running on {args.host}:{args.port}"
    )


@app.on_event("shutdown")
async def shutdown():
    global engine
    if engine:
        await engine.stop()


@app.get("/health")
async def health():
    """Health check with capacity info for upstream load balancers."""
    status = {"status": "ok"}
    if engine:
        status["capacity"] = engine.get_capacity()
    return status


@app.get("/metrics")
async def metrics():
    """Return current KV cache, prefix cache, and scheduler metrics."""
    if not engine or not engine.metrics_collector:
        return {"error": "metrics not available"}
    return engine.metrics_collector.snapshot()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": engine.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Admission control: reject if overloaded
    if engine:
        cap = engine.get_capacity()
        if not cap["accepting"]:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Server overloaded",
                    "capacity": cap,
                    "retry_after_ms": 500,
                },
            )

    request_id = tracker.add_request(request_id="")  # will be set below

    # Build sampling params
    sp = SamplingParams(
        temperature=max(request.temperature, 0.01),
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
    )

    # Submit to engine with priority
    req_id = await engine.add_request(request.prompt, sp, priority=request.priority)
    tracker._requests.pop(request_id.request_id, None)  # remove temp
    info = tracker.add_request(req_id)

    if request.stream:
        return StreamingResponse(
            _stream_generator(req_id, request),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all tokens
    full_text = []
    num_tokens = 0
    async for token_text, _ in engine.generate_stream(req_id):
        full_text.append(token_text)
        num_tokens += 1

    prompt_tokens = len(engine.tokenizer.encode(request.prompt))
    tracker.mark_finished(req_id, num_tokens)

    return CompletionResponse(
        id=req_id,
        model=engine.model_name,
        choices=[
            CompletionChoice(text="".join(full_text), finish_reason="stop")
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=num_tokens,
            total_tokens=prompt_tokens + num_tokens,
        ),
    )


async def _stream_generator(request_id: str, request: CompletionRequest):
    """SSE streaming generator for a single request."""
    rid = request_id

    def build_chunk(text: str, finish_reason: str | None) -> dict:
        return StreamingChunk(
            id=rid,
            model=engine.model_name,
            choices=[StreamingChoice(text=text, finish_reason=finish_reason)],
        ).model_dump()

    is_first = True
    async for token_text, _ in engine.generate_stream(rid):
        if is_first:
            tracker.mark_first_token(rid)
            is_first = False
        else:
            tracker.mark_token(rid)
        yield format_sse(json.dumps(build_chunk(token_text, None)))

    tracker.mark_finished(rid)
    yield format_sse(json.dumps(build_chunk("", "stop")))
    yield sse_data_end()


def main():
    parser = argparse.ArgumentParser(description="NanoServe API Server")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--scheduler", type=str, default=None,
        choices=["builtin", "fcfs", "decode_first", "chunked_prefill", "slo_aware", "priority"],
        help="Scheduler to use. Default: None (uses builtin).",
    )
    parser.add_argument(
        "--max-prefill-chunk-size", type=int, default=512,
        help="Max tokens per prefill chunk (for chunked_prefill scheduler).",
    )
    parser.add_argument(
        "--min-prefill-chunk-size", type=int, default=64,
        help="Min tokens per prefill chunk (for chunked_prefill scheduler).",
    )
    parser.add_argument(
        "--max-queue-depth", type=int, default=256,
        help="Max concurrent requests before admission control rejects with 429.",
    )
    args = parser.parse_args()

    app.state.args = args

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

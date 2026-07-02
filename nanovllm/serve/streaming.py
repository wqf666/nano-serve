"""SSE (Server-Sent Events) streaming utilities for FastAPI."""
import json
from typing import AsyncGenerator


def format_sse(data: str, event: str | None = None) -> str:
    """Format a string as an SSE message."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {data}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def sse_data_end() -> str:
    """SSE end-of-stream marker."""
    return format_sse("[DONE]")


async def streaming_response(
    token_generator: AsyncGenerator[str, None],
    chunk_builder: callable,
) -> AsyncGenerator[str, None]:
    """Convert a token generator into SSE-formatted streaming response.

    Args:
        token_generator: async generator yielding (token_text, is_end) tuples
        chunk_builder: callable(token_text, finish_reason) -> dict for SSE payload
    """
    async for token_text, is_end in token_generator:
        if is_end:
            yield format_sse(json.dumps(chunk_builder("", "stop")))
            yield sse_data_end()
        else:
            yield format_sse(json.dumps(chunk_builder(token_text, None)))

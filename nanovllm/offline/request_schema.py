"""Request and result schemas for offline batch inference."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OfflineRequest:
    """A single offline inference request.

    Attributes:
        id: Unique request identifier (used for ordering and dedup).
        prompt: Text prompt or list of token IDs.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (default 0.6).
        prefix_key: Optional grouping key for prefix-aware batching.
        metadata: Arbitrary extra metadata.
    """
    id: str
    prompt: str | list[int]
    max_tokens: int = 256
    temperature: float = 0.6
    prefix_key: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if isinstance(self.prompt, str):
            d["prompt"] = self.prompt
        else:
            d["prompt_token_ids"] = self.prompt
        if self.prefix_key:
            d["prefix_key"] = self.prefix_key
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class OfflineResult:
    """Result for a single offline request.

    Attributes:
        id: Matching request ID.
        text: Generated text.
        token_ids: Generated token IDs.
        prompt_tokens: Number of prompt tokens.
        output_tokens: Number of generated tokens.
        latency_s: Time from submission to completion (seconds).
        error: Error message if the request failed.
    """
    id: str
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 6),
            "error": self.error,
        }

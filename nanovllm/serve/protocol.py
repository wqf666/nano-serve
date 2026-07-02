"""Request/Response schemas for NanoServe online serving."""
import time
from typing import Optional
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Optional[str] = None


class StreamingChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: float = Field(default_factory=time.time)
    model: str = ""
    choices: list[CompletionChoice] = []
    usage: UsageInfo = UsageInfo()


class StreamingChunk(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: float = Field(default_factory=time.time)
    model: str = ""
    choices: list[StreamingChoice] = []

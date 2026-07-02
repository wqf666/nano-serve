"""Request tracking with timestamps for metrics collection."""
import time
from dataclasses import dataclass, field


@dataclass
class RequestInfo:
    request_id: str
    arrival_time: float = field(default_factory=time.time)
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_time: float | None = None
    num_prompt_tokens: int = 0
    num_completion_tokens: int = 0
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        """Time to first token."""
        if self.first_token_time:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def e2e_latency(self) -> float | None:
        """End-to-end latency."""
        if self.finish_time:
            return self.finish_time - self.arrival_time
        return None


class RequestTracker:
    def __init__(self):
        self._requests: dict[str, RequestInfo] = {}

    def add_request(self, request_id: str, num_prompt_tokens: int = 0) -> RequestInfo:
        info = RequestInfo(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
        )
        self._requests[request_id] = info
        return info

    def mark_first_token(self, request_id: str):
        if request_id in self._requests:
            info = self._requests[request_id]
            if info.first_token_time is None:
                info.first_token_time = time.time()
            info.last_token_time = time.time()

    def mark_token(self, request_id: str):
        if request_id in self._requests:
            self._requests[request_id].last_token_time = time.time()

    def mark_finished(self, request_id: str, num_completion_tokens: int = 0):
        if request_id in self._requests:
            info = self._requests[request_id]
            info.finish_time = time.time()
            info.num_completion_tokens = num_completion_tokens

    def mark_error(self, request_id: str, error: str):
        if request_id in self._requests:
            info = self._requests[request_id]
            info.finish_time = time.time()
            info.error = error

    def get_info(self, request_id: str) -> RequestInfo | None:
        return self._requests.get(request_id)

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._requests.values() if r.finish_time is None)

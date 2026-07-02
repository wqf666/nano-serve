from nanovllm.scheduler.base import BaseScheduler, SchedulingBudget, RequestState
from nanovllm.scheduler.fcfs import FCFSScheduler
from nanovllm.scheduler.decode_first import DecodeFirstScheduler
from nanovllm.scheduler.chunked_prefill import ChunkedPrefillScheduler

SCHEDULER_REGISTRY: dict[str, type[BaseScheduler]] = {
    "fcfs": FCFSScheduler,
    "decode_first": DecodeFirstScheduler,
    "chunked_prefill": ChunkedPrefillScheduler,
}


def create_scheduler(name: str, config, **kwargs) -> BaseScheduler:
    """Factory: create a scheduler by name with optional extra params."""
    cls = SCHEDULER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            f"Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
    return cls(config, **kwargs)

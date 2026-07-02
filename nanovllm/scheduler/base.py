"""Base scheduler interface, request state machine, and scheduling budget.

All schedulers (FCFS, Decode-first, Chunked Prefill, SLO-aware) must
subclass BaseScheduler and implement the four abstract methods.

The scheduler is called by the engine step loop:
    1. engine.add_request() → scheduler.add(seq)
    2. engine.step() → seqs, is_prefill = scheduler.schedule()
                     → model_runner(seqs, is_prefill)
                     → scheduler.postprocess(seqs, token_ids, is_prefill)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto


class RequestState(Enum):
    """Request lifecycle states.

    Phase 3 uses WAITING / RUNNING / FINISHED (matching the built-in scheduler).
    Phase 4+ will add WAITING_DECODE / RUNNING_DECODE / PREEMPTED.
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    # Future states (Phase 4+):
    # WAITING_DECODE = auto()
    # RUNNING_DECODE = auto()
    # PREEMPTED = auto()


@dataclass
class SchedulingBudget:
    """Token and sequence budget for a single scheduling step.

    Used by Phase 4+ schedulers (Decode-first, Chunked Prefill) to make
    token-budget-aware decisions. Phase 3 FCFS uses max_num_seqs and
    max_num_batched_tokens from config directly.
    """
    max_num_seqs: int
    max_num_batched_tokens: int
    max_num_prefill_tokens: int = 0
    max_num_decode_tokens: int = 0


class BaseScheduler(ABC):
    """Abstract base class for all NanoServe schedulers.

    Subclasses must implement:
        schedule()          — select sequences for this step
        postprocess()       — update sequences after model forward
        add()               — enqueue a new request
        is_finished()       — check if all requests are done

    The scheduler manages two deques:
        self.waiting  — sequences awaiting prefill
        self.running  — sequences in decode phase
    """

    @abstractmethod
    def schedule(self) -> tuple[list, bool]:
        """Select sequences for the current step.

        Returns:
            (scheduled_seqs, is_prefill)
            is_prefill=True means this step runs prefill for the batch
            is_prefill=False means this step runs decode for the batch
        """
        ...

    @abstractmethod
    def postprocess(self, seqs: list, token_ids: list[int], is_prefill: bool):
        """Update sequences after model forward pass.

        Responsibilities:
        - Hash blocks for prefix caching
        - Update cached token counts
        - Append generated tokens
        - Mark finished sequences (EOS or max_tokens)
        - Deallocate finished sequences' blocks
        """
        ...

    @abstractmethod
    def add(self, seq):
        """Add a new sequence to the scheduler."""
        ...

    @abstractmethod
    def is_finished(self) -> bool:
        """Return True if no waiting or running sequences remain."""
        ...

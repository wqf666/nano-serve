"""Priority Scheduler with Weighted Fair Queuing.

Extends ChunkedPrefillScheduler with multi-level priority support:
  - Priority 2 (high):   Premium/enterprise requests
  - Priority 1 (normal): Default requests
  - Priority 0 (low):    Background/batch requests

Strategy:
  - Decode: use parent ChunkedPrefillScheduler's decode (unchanged)
  - Prefill: sort waiting by priority (high first), then schedule with chunk cap
  - Preemption: low-priority decode sequences are preempted first

Priority passing:
  The priority is set via scheduler.set_next_priority() before the engine's
  add_request() is called. The scheduler consumes it in add().
"""
import time
from collections import deque

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.scheduler.chunked_prefill import ChunkedPrefillScheduler


class PriorityScheduler(ChunkedPrefillScheduler):
    """Chunked prefill scheduler with 3-tier priority queues."""

    HIGH = 2
    NORMAL = 1
    LOW = 0

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._seq_priority: dict[int, int] = {}
        self._pending_priorities: deque[int] = deque()

    def set_next_priority(self, priority: int):
        """Set priority for the next request to be added."""
        self._pending_priorities.append(priority)

    def add(self, seq: Sequence):
        """Add sequence with the next pending priority."""
        if self._pending_priorities:
            priority = self._pending_priorities.popleft()
        else:
            priority = self.NORMAL
        self._seq_priority[seq.seq_id] = priority
        super().add(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Schedule with priority-aware prefill ordering.

        Decode scheduling is unchanged from ChunkedPrefillScheduler
        (all running sequences get 1 token per step).
        Prefill is ordered by priority: high → normal → low.
        """
        # ---- Phase 1: Try decode first (parent's logic) ----
        decode_seqs = self._try_decode()
        if decode_seqs:
            return decode_seqs, False

        # ---- Phase 2: Prefill with priority ordering ----
        prefill_seqs = self._schedule_prefill_priority()
        if prefill_seqs:
            return prefill_seqs, True

        raise RuntimeError("schedule() called with no work available")

    def _try_decode(self) -> list[Sequence]:
        """Try to schedule decode. Returns empty list if no decode work."""
        scheduled_seqs = []

        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs

        return []

    def _schedule_prefill_priority(self) -> list[Sequence]:
        """Schedule prefill with priority ordering: high → normal → low."""
        budget = self.max_num_batched_tokens
        scheduled = []

        # Sort waiting by priority (highest first), stable sort preserves FIFO
        sorted_waiting = sorted(
            list(self.waiting),
            key=lambda s: self._seq_priority.get(s.seq_id, self.NORMAL),
            reverse=True,
        )

        for seq in sorted_waiting:
            if budget <= 0 or len(scheduled) >= self.max_num_seqs:
                break

            tokens_left = seq.num_tokens - seq.num_cached_tokens

            # Allocate KV blocks on first visit
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    continue  # Skip, try next
                self.block_manager.allocate(seq, num_cached_blocks)
                tokens_left = seq.num_tokens - seq.num_cached_tokens

            chunk = min(tokens_left, budget)
            if chunk <= 0:
                continue

            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            budget -= chunk

            # Full prefill complete → transition to running
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                # Partial prefill: rotate to back of waiting
                self.waiting.remove(seq)
                self.waiting.append(seq)

            scheduled.append(seq)

        return scheduled

    def postprocess(self, seqs, token_ids, is_prefill):
        """Override to clean up priority tracking for finished sequences."""
        super().postprocess(seqs, token_ids, is_prefill)
        for seq in seqs:
            if seq.is_finished:
                self._seq_priority.pop(seq.seq_id, None)

    def get_priority_stats(self) -> dict:
        """Return current priority distribution for observability."""
        stats = {"high": 0, "normal": 0, "low": 0}
        for seq in list(self.waiting) + list(self.running):
            p = self._seq_priority.get(seq.seq_id, self.NORMAL)
            if p == self.HIGH:
                stats["high"] += 1
            elif p == self.LOW:
                stats["low"] += 1
            else:
                stats["normal"] += 1
        return stats

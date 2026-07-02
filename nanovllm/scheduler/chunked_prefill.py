"""Chunked Prefill Scheduler (v2 — prefill-first with chunk size cap).

Root cause of v1 regression:
    v1 used decode-first ordering (same as Phase 4 DecodeFirstScheduler),
    causing prefill starvation. New requests waited behind continuous decode
    steps, inflating TTFT from 85ms to 2153ms.

v2 fix:
    Use PREFILL-FIRST ordering (same as FCFS) with a per-sequence chunk
    size cap. This preserves FCFS behavior for short prompts while splitting
    long prompts across multiple prefill steps.

Strategy:
    1. Try prefill first — schedule waiting sequences with chunk cap.
    2. If no prefill work, schedule all running decode sequences.
    3. Long prompts are rotated to back of waiting queue after partial
       prefill (fairness guard: shorter prompts get a chance next round).

Trade-off vs FCFS:
    - Short prompts (< chunk size): identical to FCFS (full prefill in 1 step).
    - Long prompts (> chunk size): split across steps, slightly longer TTFT
      for the long prompt itself, but decode sequences get steps back sooner.
    - Overall: TTFT and ITL should match FCFS for typical workloads,
      with improved head-of-line blocking for long prompts.
"""
import logging

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.scheduler.fcfs import FCFSScheduler

logger = logging.getLogger(__name__)


class ChunkedPrefillScheduler(FCFSScheduler):
    """Prefill-first scheduler with per-sequence chunk size cap.

    Inherits postprocess, preempt, block management from FCFSScheduler.
    """

    def __init__(self, config, max_prefill_chunk_size: int = 512,
                 min_prefill_chunk_size: int = 64, **kwargs):
        super().__init__(config)
        self.max_prefill_chunk = max_prefill_chunk_size
        self.min_prefill_chunk = min_prefill_chunk_size
        # Step-level trace counters
        self._step_count = 0
        self._trace_log: list[dict] = []

    def schedule(self) -> tuple[list[Sequence], bool]:
        self._step_count += 1
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---- Phase 1: PREFILL FIRST (same priority as FCFS) ----
        # Cap each sequence's prefill to max_prefill_chunk tokens.
        # Rotate partially-prefilled sequences for fairness.
        attempts = 0
        max_attempts = len(self.waiting) + 1  # prevent infinite loop

        while (
            self.waiting
            and len(scheduled_seqs) < self.max_num_seqs
            and attempts < max_attempts
        ):
            attempts += 1
            seq = self.waiting[0]
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            if remaining_budget <= 0:
                break

            # How many tokens still need prefill for this sequence
            tokens_left = seq.num_tokens - seq.num_cached_tokens

            # Allocate KV blocks on first visit
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # Not enough KV blocks
                self.block_manager.allocate(seq, num_cached_blocks)
                tokens_left = seq.num_tokens - seq.num_cached_tokens

            # Apply chunk cap
            chunk = min(tokens_left, remaining_budget, self.max_prefill_chunk)
            if chunk < self.min_prefill_chunk and scheduled_seqs:
                break  # Too small for an efficient step

            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            num_batched_tokens += chunk

            # Check if prefill is complete for this sequence
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                # Full prefill → transition to running (decode)
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                # Partial prefill → rotate to back for fairness
                self.waiting.popleft()
                self.waiting.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self._log_trace("prefill", scheduled_seqs, num_batched_tokens)
            return scheduled_seqs, True

        # ---- Phase 2: DECODE (only when no prefill work exists) ----
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
            self._log_trace("decode", scheduled_seqs, len(scheduled_seqs))
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False

        raise RuntimeError("schedule() called with no work available")

    def _log_trace(self, step_type: str, seqs: list, tokens: int):
        """Record step-level trace for debugging and analysis."""
        trace = {
            "step": self._step_count,
            "type": step_type,
            "num_seqs": len(seqs),
            "tokens": tokens,
            "waiting": len(self.waiting),
            "running": len(self.running),
        }
        self._trace_log.append(trace)

        # Keep only last 1000 entries
        if len(self._trace_log) > 1000:
            self._trace_log = self._trace_log[-500:]

    def get_trace(self) -> list[dict]:
        """Return the step-level trace log."""
        return list(self._trace_log)

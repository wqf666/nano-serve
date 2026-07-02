"""Decode-first Scheduler.

Strategy: prioritize decode (ongoing generation) over prefill (new prompts).
    1. Schedule ALL running decode requests first
    2. If token budget remains, schedule waiting prefill requests
    3. No chunked prefill in this phase

Trade-off vs FCFS:
    - ITL (inter-token latency) should improve because decode is never
      delayed by long prefill operations in the same step.
    - TTFT (time to first token) may increase because new requests wait
      longer while decode requests are served first.
    - Overall throughput should remain similar since total tokens per step
      is bounded by max_num_batched_tokens, not scheduling order.
"""
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.scheduler.fcfs import FCFSScheduler


class DecodeFirstScheduler(FCFSScheduler):
    """Decode-first scheduler: decode requests always run before prefill.

    Inherits postprocess, preempt, block management from FCFSScheduler.
    Only overrides schedule() to reverse the priority order.
    """

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---- Phase 1: Decode — schedule ALL running sequences first ----
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
                num_batched_tokens += 1

        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False

        # ---- Phase 2: Prefill — only if no decode work exists ----
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Allow chunked prefill only for the first seq in a batch
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # Should not reach here if called correctly
        raise RuntimeError("schedule() called with no work available")

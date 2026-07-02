"""FCFS (First-Come, First-Served) Scheduler.

This is a faithful migration of the built-in engine/scheduler.py logic
into the BaseScheduler interface. The scheduling strategy:
    1. Try to schedule waiting (prefill) sequences first, FIFO order.
    2. If no prefill work, schedule running (decode) sequences.
    3. Preempt running sequences when KV cache is exhausted.

No new scheduling behavior is introduced — this is the baseline scheduler
for comparing against Decode-first, Chunked Prefill, and SLO-aware.
"""
from collections import deque

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.scheduler.base import BaseScheduler


class FCFSScheduler(BaseScheduler):
    """First-Come, First-Served scheduler.

    Migrated from the built-in nanovllm.engine.scheduler.Scheduler.
    """

    def __init__(self, config, **kwargs):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---- Prefill phase: try waiting sequences in FIFO order ----
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # Determine how many tokens this prefill needs
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # Not enough KV blocks available
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Only allow chunked prefill for the first seq in a batch
            if remaining < num_tokens and scheduled_seqs:
                break

            # Allocate KV blocks if needed
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # If full prompt is scheduled, move to running
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # ---- Decode phase: schedule all running sequences ----
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            # Ensure KV cache blocks are available for next token
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

        assert scheduled_seqs, "schedule() called with no work"
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """Preempt a running sequence back to waiting."""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ):
        """Update sequences after model forward pass."""
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # Chunked prefill: skip token append if prompt not fully processed
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)

            # Check termination: EOS or max_tokens
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

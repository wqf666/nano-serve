"""SLO-aware Scheduler.

Strategy: two-level priority based on SLO violation risk.

    SLO definitions:
      - TTFT SLO: time from request arrival to first output token
      - ITL SLO: time between consecutive output tokens during decode

    Each step:
      1. Separate decode sequences into high-priority (ITL urgent) and normal
      2. Separate prefill sequences into high-priority (TTFT urgent) and normal
      3. Schedule order: high-priority decode → normal decode → high-priority prefill → normal prefill
      4. Starvation guard: force high-priority for sequences waiting too long

    SLO targets auto-calibrated from first N requests:
      target_ttft = initial_p50_ttft * multiplier
      target_itl  = initial_p50_itl  * multiplier
"""
import time
import logging
from collections import deque

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.scheduler.fcfs import FCFSScheduler

logger = logging.getLogger(__name__)


class SLOAwareScheduler(FCFSScheduler):
    """SLO-aware scheduler with two-level priority queues.

    Inherits postprocess, preempt, block management from FCFSScheduler.
    """

    def __init__(self, config, target_ttft: float = 0.0,
                 target_itl: float = 0.0, slo_multiplier: float = 1.5,
                 starvation_timeout: float = 10.0, **kwargs):
        super().__init__(config)
        self.target_ttft = target_ttft       # seconds, 0 = auto-calibrate
        self.target_itl = target_itl         # seconds, 0 = auto-calibrate
        self.slo_multiplier = slo_multiplier
        self.starvation_timeout = starvation_timeout

        # Per-sequence timing tracking
        self._enqueue_times: dict[int, float] = {}    # seq_id → arrival time
        self._last_decode_times: dict[int, float] = {}  # seq_id → last decode step time

        # Auto-calibration data
        self._calibration_ttfts: list[float] = []
        self._calibration_itls: list[float] = []
        self._calibrated = False
        self._calibration_target = 20  # calibrate after N completions

    def add(self, seq: Sequence):
        """Track enqueue time for TTFT SLO monitoring."""
        super().add(seq)
        self._enqueue_times[seq.seq_id] = time.time()

    def schedule(self) -> tuple[list[Sequence], bool]:
        now = time.time()

        # ---- Try decode first (with priority split) ----
        hi_decode = []
        lo_decode = []
        remaining = list(self.running)
        self.running.clear()

        for seq in remaining:
            if not self.block_manager.can_append(seq):
                # Need preemption — handle below
                self.running.append(seq)
                continue

            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)

            # Check ITL urgency
            last_decode = self._last_decode_times.get(seq.seq_id, now)
            itl_elapsed = now - last_decode
            is_urgent = (
                (self.target_itl > 0 and itl_elapsed > self.target_itl * 0.8)
                or self._is_starving(seq.seq_id, now)
            )

            if is_urgent:
                hi_decode.append(seq)
            else:
                lo_decode.append(seq)

        # Handle preemption for sequences that couldn't be scheduled
        while self.running:
            seq = self.running.popleft()
            if not self.block_manager.can_append(seq):
                # Preempt: prefer preempting low-priority first
                if lo_decode:
                    self.preempt(lo_decode.pop())
                    lo_decode.append(seq)
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append(seq)
                else:
                    self.preempt(seq)

        # Combine decode sequences: high priority first
        decode_seqs = hi_decode + lo_decode

        if decode_seqs:
            for seq in decode_seqs:
                self._last_decode_times[seq.seq_id] = now
            self.running.extend(decode_seqs)
            return decode_seqs, False

        # ---- Prefill (with priority split) ----
        hi_prefill = []
        lo_prefill = []
        budget = self.max_num_batched_tokens
        attempts = 0
        max_attempts = len(self.waiting) + 1

        while self.waiting and budget > 0 and attempts < max_attempts:
            attempts += 1
            seq = self.waiting[0]

            tokens_left = seq.num_tokens - seq.num_cached_tokens

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)
                tokens_left = seq.num_tokens - seq.num_cached_tokens

            chunk = min(tokens_left, budget)
            if chunk <= 0:
                break

            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            budget -= chunk

            # Check TTFT urgency
            enqueue_time = self._enqueue_times.get(seq.seq_id, now)
            ttft_elapsed = now - enqueue_time
            is_urgent = (
                (self.target_ttft > 0 and ttft_elapsed > self.target_ttft * 0.8)
                or self._is_starving(seq.seq_id, now)
            )

            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                self.waiting.popleft()
                self.waiting.append(seq)

            if is_urgent:
                hi_prefill.append(seq)
            else:
                lo_prefill.append(seq)

        prefill_seqs = hi_prefill + lo_prefill
        if prefill_seqs:
            return prefill_seqs, True

        raise RuntimeError("schedule() called with no work available")

    def postprocess(self, seqs, token_ids, is_prefill):
        """Override to track decode timing and auto-calibrate SLOs."""
        super().postprocess(seqs, token_ids, is_prefill)

        now = time.time()
        if not is_prefill:
            for seq in seqs:
                self._last_decode_times[seq.seq_id] = now

        # Auto-calibrate from completed sequences
        if not self._calibrated:
            for seq in seqs:
                if seq.is_finished:
                    # TTFT: time from enqueue to first decode
                    enqueue_t = self._enqueue_times.get(seq.seq_id)
                    first_decode_t = self._last_decode_times.get(seq.seq_id)
                    if enqueue_t and first_decode_t:
                        self._calibration_ttfts.append(first_decode_t - enqueue_t)

                    # ITL: average time between decode steps
                    # (approximated by total decode time / num_completion_tokens)
                    if seq.num_completion_tokens > 1:
                        total_decode_time = now - enqueue_t if enqueue_t else 0
                        avg_itl = total_decode_time / seq.num_completion_tokens
                        self._calibration_itls.append(avg_itl)

                    # Clean up
                    self._enqueue_times.pop(seq.seq_id, None)
                    self._last_decode_times.pop(seq.seq_id, None)

            if len(self._calibration_ttfts) >= self._calibration_target:
                self._auto_calibrate()

        # Clean up tracking for finished sequences
        for seq in seqs:
            if seq.is_finished:
                self._enqueue_times.pop(seq.seq_id, None)
                self._last_decode_times.pop(seq.seq_id, None)

    def _auto_calibrate(self):
        """Set SLO targets from observed baseline metrics."""
        if not self._calibration_ttfts or not self._calibration_itls:
            return
        sorted_ttfts = sorted(self._calibration_ttfts)
        sorted_itls = sorted(self._calibration_itls)
        p50_idx = len(sorted_ttfts) // 2

        self.target_ttft = sorted_ttfts[p50_idx] * self.slo_multiplier
        self.target_itl = sorted_itls[p50_idx] * self.slo_multiplier
        self._calibrated = True

        logger.info(
            f"SLO auto-calibrated: target_ttft={self.target_ttft:.4f}s, "
            f"target_itl={self.target_itl:.4f}s "
            f"(from {len(self._calibration_ttfts)} samples, "
            f"multiplier={self.slo_multiplier})"
        )

    def _is_starving(self, seq_id: int, now: float) -> bool:
        """Check if a sequence has been waiting too long."""
        enqueue_time = self._enqueue_times.get(seq_id, now)
        return (now - enqueue_time) > self.starvation_timeout

    def get_slo_targets(self) -> dict:
        """Return current SLO targets."""
        return {
            "target_ttft": self.target_ttft,
            "target_itl": self.target_itl,
            "calibrated": self._calibrated,
            "calibration_samples": len(self._calibration_ttfts),
        }

"""Metrics collector for KV cache, prefix cache, and scheduler observability.

Design:
    MetricsCollector monkey-patches BlockManager.allocate() and
    BlockManager.deallocate() to record prefix cache hit/miss stats
    WITHOUT modifying core engine files.

    It also provides a snapshot() method that reads the current state
    of the BlockManager and Scheduler to produce a metrics dict.

Usage:
    collector = MetricsCollector()
    collector.attach(block_manager, scheduler)
    # ... engine runs ...
    snapshot = collector.snapshot()  # returns dict of current metrics
    trace = collector.get_trace()    # returns list of step-level traces
"""
import time
from collections import deque


class MetricsCollector:
    """Collects KV cache, prefix cache, and scheduler metrics."""

    def __init__(self):
        self.block_manager = None
        self.scheduler = None

        # Cumulative prefix cache stats
        self.prefix_cache_hits = 0      # blocks reused from cache
        self.prefix_cache_misses = 0    # blocks freshly allocated
        self.reused_prefix_tokens = 0   # tokens saved by prefix cache

        # Per-step stats (reset each step)
        self.freed_blocks_per_step = 0
        self.allocated_blocks_per_step = 0

        # Step-level trace log
        self._step_count = 0
        self._trace_log = deque(maxlen=2000)

        # Snapshot of last step's scheduling info
        self._last_scheduled_prefill_tokens = 0
        self._last_scheduled_decode_tokens = 0

    def attach(self, block_manager, scheduler):
        """Monkey-patch BlockManager methods to record metrics."""
        self.block_manager = block_manager
        self.scheduler = scheduler

        original_allocate = block_manager.allocate
        original_deallocate = block_manager.deallocate

        def traced_allocate(seq, num_cached_blocks):
            # Record prefix cache stats
            if num_cached_blocks > 0:
                self.prefix_cache_hits += num_cached_blocks
                self.reused_prefix_tokens += num_cached_blocks * block_manager.block_size
            new_blocks = seq.num_blocks - num_cached_blocks
            self.prefix_cache_misses += new_blocks
            self.allocated_blocks_per_step += new_blocks
            return original_allocate(seq, num_cached_blocks)

        def traced_deallocate(seq):
            num_blocks = len(seq.block_table)
            self.freed_blocks_per_step += num_blocks
            return original_deallocate(seq)

        block_manager.allocate = traced_allocate
        block_manager.deallocate = traced_deallocate

    def record_step(self, is_prefill: bool, num_seqs: int, num_tokens: int):
        """Record a step's scheduling info. Called by OnlineEngine.step()."""
        self._step_count += 1

        if is_prefill:
            self._last_scheduled_prefill_tokens = num_tokens
            self._last_scheduled_decode_tokens = 0
        else:
            self._last_scheduled_prefill_tokens = 0
            self._last_scheduled_decode_tokens = num_tokens

        # Build step trace
        bm = self.block_manager
        sched = self.scheduler
        total_blocks = len(bm.blocks)
        used_blocks = len(bm.used_block_ids)
        free_blocks = len(bm.free_block_ids)
        cached_but_free = sum(
            1 for bid in bm.free_block_ids
            if bm.blocks[bid].hash != -1
        )

        trace = {
            "step": self._step_count,
            "timestamp": time.time(),
            "active_requests": len(sched.running),
            "waiting_requests": len(sched.waiting),
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "total_blocks": total_blocks,
            "block_utilization": round(used_blocks / max(total_blocks, 1), 4),
            "cached_but_free_blocks": cached_but_free,
            "prefix_cache_hit_rate": self._hit_rate(),
            "scheduled_prefill_tokens": self._last_scheduled_prefill_tokens,
            "scheduled_decode_tokens": self._last_scheduled_decode_tokens,
            "allocated_blocks_this_step": self.allocated_blocks_per_step,
            "freed_blocks_this_step": self.freed_blocks_per_step,
        }
        self._trace_log.append(trace)

        # Reset per-step counters
        self.allocated_blocks_per_step = 0
        self.freed_blocks_per_step = 0

    def _hit_rate(self) -> float:
        total = self.prefix_cache_hits + self.prefix_cache_misses
        if total == 0:
            return 0.0
        return round(self.prefix_cache_hits / total, 4)

    def snapshot(self) -> dict:
        """Return current metrics snapshot."""
        bm = self.block_manager
        sched = self.scheduler

        if not bm or not sched:
            return {"error": "MetricsCollector not attached"}

        total_blocks = len(bm.blocks)
        used_blocks = len(bm.used_block_ids)
        free_blocks = len(bm.free_block_ids)
        cached_but_free = sum(
            1 for bid in bm.free_block_ids
            if bm.blocks[bid].hash != -1
        )
        hash_entries = len(bm.hash_to_block_id)

        return {
            "kv_cache": {
                "total_blocks": total_blocks,
                "used_blocks": used_blocks,
                "free_blocks": free_blocks,
                "block_utilization": round(used_blocks / max(total_blocks, 1), 4),
                "cached_but_free_blocks": cached_but_free,
                "hash_table_entries": hash_entries,
            },
            "prefix_cache": {
                "hits": self.prefix_cache_hits,
                "misses": self.prefix_cache_misses,
                "hit_rate": self._hit_rate(),
                "reused_prefix_tokens": self.reused_prefix_tokens,
                "saved_prefill_tokens": self.reused_prefix_tokens,
            },
            "scheduler": {
                "active_requests": len(sched.running),
                "waiting_requests": len(sched.waiting),
                "total_steps": self._step_count,
                "last_prefill_tokens": self._last_scheduled_prefill_tokens,
                "last_decode_tokens": self._last_scheduled_decode_tokens,
            },
        }

    def get_trace(self) -> list[dict]:
        """Return step-level trace log."""
        return list(self._trace_log)

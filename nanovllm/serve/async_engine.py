"""AsyncEngineLoop: wraps nano-vllm's synchronous engine into an async serving loop.

Design:
- OnlineEngine subclasses LLMEngine, overriding step() to also return
  per-sequence newly generated token IDs (needed for streaming).
- AsyncEngine runs OnlineEngine + a background asyncio task that calls
  step() in a thread executor (step is blocking on CUDA).
- Per-request asyncio.Queue delivers tokens to individual HTTP handlers.
"""
import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class OnlineEngine(LLMEngine):
    """LLMEngine subclass that exposes per-step token info for streaming.

    Accepts an optional scheduler_name parameter to swap the built-in
    scheduler with a BaseScheduler implementation (FCFS, Decode-first, etc.).
    """

    def __init__(self, model: str, scheduler_name: str | None = None,
                 scheduler_params: dict | None = None, **kwargs):
        # Build config early so we can pass it to the scheduler factory
        from dataclasses import fields as dc_fields
        from nanovllm.config import Config
        config_fields = {field.name for field in dc_fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)

        super().__init__(model, **kwargs)
        self._scheduler_name = scheduler_name or "builtin"

        # Sync eos from tokenizer into our config copy (set after tokenizer load)
        self.config.eos = self.tokenizer.eos_token_id

        # Swap scheduler if a named scheduler is requested
        if scheduler_name and scheduler_name != "builtin":
            from nanovllm.scheduler import create_scheduler
            extra = scheduler_params or {}
            new_scheduler = create_scheduler(scheduler_name, self.config, **extra)

            # Transfer state from the old scheduler:
            # - block_manager (already initialized with actual num_kvcache_blocks)
            # - waiting/running deques (usually empty at init)
            old_sched = self.scheduler
            new_scheduler.block_manager = old_sched.block_manager
            new_scheduler.waiting = old_sched.waiting
            new_scheduler.running = old_sched.running
            new_scheduler.eos = old_sched.eos
            new_scheduler.block_size = old_sched.block_size

            self.scheduler = new_scheduler
            logger.info(f"Scheduler swapped to: {scheduler_name}")

    def step(self):
        """Run one engine step.

        Returns:
            finished_outputs: list of (seq_id, all_completion_token_ids)
            num_tokens: positive for prefill, negative for decode
            new_tokens: dict[seq_id, list[int]] — newly generated token IDs
                        for ALL sequences (both running and just-finished)
        """
        # Snapshot completion token counts before the step
        prev_counts: dict[int, int] = {}
        for seq in self.scheduler.running:
            prev_counts[seq.seq_id] = seq.num_completion_tokens

        # Execute the standard step (schedule → model forward → postprocess)
        finished_outputs, num_tokens = super().step()

        # After step: extract newly generated tokens per sequence
        new_tokens: dict[int, list[int]] = {}

        # Running sequences (still alive after postprocess)
        for seq in self.scheduler.running:
            prev = prev_counts.get(seq.seq_id, 0)
            curr = seq.num_completion_tokens
            if curr > prev:
                new_tokens[seq.seq_id] = list(seq.completion_token_ids[prev:curr])

        # Just-finished sequences (removed from running by postprocess)
        for seq_id, token_ids in finished_outputs:
            prev = prev_counts.get(seq_id, 0)
            if len(token_ids) > prev:
                new_tokens[seq_id] = list(token_ids[prev:])

        return finished_outputs, num_tokens, new_tokens


class AsyncEngine:
    """Async wrapper around OnlineEngine with background step loop."""

    def __init__(self, model: str, scheduler_name: str | None = None,
                 scheduler_params: dict | None = None, **kwargs):
        self.engine = OnlineEngine(
            model, scheduler_name=scheduler_name,
            scheduler_params=scheduler_params, **kwargs
        )
        self.model_name = model.rstrip("/").split("/")[-1]
        self.scheduler_name = self.engine._scheduler_name
        self.tokenizer = self.engine.tokenizer
        self._request_counter = 0
        self._token_queues: dict[str, asyncio.Queue] = {}
        self._seq_to_request: dict[int, str] = {}  # seq_id → request_id
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._loop_task: asyncio.Task | None = None
        self._pending_add: deque[tuple[str, str, SamplingParams]] = deque()

    async def start(self):
        """Start the background engine loop."""
        self._loop_task = asyncio.create_task(self._engine_loop())

    async def stop(self):
        """Stop the background engine loop."""
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)
        self.engine.exit()

    async def add_request(
        self, prompt: str, sampling_params: SamplingParams
    ) -> str:
        """Submit a generation request. Returns request_id."""
        self._request_counter += 1
        request_id = f"req-{self._request_counter}-{time.time():.6f}"
        self._token_queues[request_id] = asyncio.Queue()
        self._pending_add.append((request_id, prompt, sampling_params))
        return request_id

    async def generate_stream(self, request_id: str):
        """Async generator yielding (token_text, is_end) tuples."""
        queue = self._token_queues.get(request_id)
        if not queue:
            return

        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=300)
                if item is None:
                    break
                yield item, False
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out after 300s")
        finally:
            self._token_queues.pop(request_id, None)

    async def _engine_loop(self):
        """Background loop: add pending requests, run step, dispatch tokens."""
        while True:
            try:
                # Drain pending add_request calls
                while self._pending_add:
                    request_id, prompt, sp = self._pending_add.popleft()

                    # add_request runs in executor (may trigger CUDA ops)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        self._executor, self.engine.add_request, prompt, sp
                    )

                    # Retrieve the seq_id of the just-added sequence
                    # (it's the last item in scheduler.waiting deque)
                    seq = self.engine.scheduler.waiting[-1]
                    self._seq_to_request[seq.seq_id] = request_id

                # Run step if there are active sequences
                has_work = (
                    self.engine.scheduler.waiting
                    or self.engine.scheduler.running
                )

                if has_work:
                    loop = asyncio.get_running_loop()
                    finished, num_tokens, new_tokens = (
                        await loop.run_in_executor(
                            self._executor, self.engine.step
                        )
                    )

                    # Dispatch new tokens to per-request queues
                    for seq_id, token_ids in new_tokens.items():
                        rid = self._seq_to_request.get(seq_id)
                        if rid and rid in self._token_queues:
                            text = self.tokenizer.decode(token_ids)
                            await self._token_queues[rid].put(text)

                    # Signal completion for finished sequences
                    for seq_id, _ in finished:
                        rid = self._seq_to_request.pop(seq_id, None)
                        if rid and rid in self._token_queues:
                            await self._token_queues[rid].put(None)

                else:
                    await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Engine loop error: {e}", exc_info=True)
                # Cancel any stuck requests
                for rid, q in list(self._token_queues.items()):
                    await q.put(None)
                self._seq_to_request.clear()
                await asyncio.sleep(0.1)

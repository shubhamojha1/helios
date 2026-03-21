import asyncio
import time
from typing import Dict, List, Optional
from core.types import Request, RequestStatus, SchedulerOutput
from memory.manager import MemoryManager
from engine.engine import Engine
from scheduler.fcfs import FCFSStrategy


# Sentinel value placed on output_queue to signal completion
STREAM_DONE = None


class Scheduler:
    def __init__(
        self,
        engine: Engine,
        memory_manager: MemoryManager,
        max_batch_size: int = 8,
    ):
        self.engine = engine
        self.memory_manager = memory_manager
        self.strategy = FCFSStrategy(max_batch_size=max_batch_size)

        # Waiting queue — requests submitted but not yet active
        self.waiting: List[Request] = []

        # Active set — requests currently being processed
        self.active: List[Request] = []

        # Lookup by request_id for cancellation
        self.all_requests: Dict[str, Request] = {}

        self._step_id = 0
        self._running = False
        self._lock = asyncio.Lock()

    async def submit(self, request: Request) -> None:
        """
        Called by the API layer to add a new request.
        Tokenizes the prompt and adds to waiting queue.
        """
        async with self._lock:
            request.prompt_tokens = self.engine.tokenize(request.prompt)
            request.status = RequestStatus.WAITING
            self.waiting.append(request)
            self.all_requests[request.request_id] = request

    async def cancel(self, request_id: str) -> bool:
        async with self._lock:
            req = self.all_requests.get(request_id)
            if req is None:
                return False
            if req.status in (RequestStatus.COMPLETE, RequestStatus.CANCELLED):
                return False

            req.status = RequestStatus.CANCELLED
            self.memory_manager.free(request_id)

            # Remove from waiting/active lists
            self.waiting = [r for r in self.waiting if r.request_id != request_id]
            self.active = [r for r in self.active if r.request_id != request_id]

            # Signal the API layer that this stream is done
            await req.output_queue.put(STREAM_DONE)
            return True

    async def run(self) -> None:
        """
        Main control loop. Runs until stop() is called.
        Each iteration: schedule -> prefill -> decode -> emit tokens.
        """
        self._running = True
        print("Scheduler running.")

        while self._running:
            async with self._lock:
                await self._step()

            # Yield control so the event loop can handle API requests
            await asyncio.sleep(0)

    def stop(self) -> None:
        self._running = False

    async def _step(self) -> None:
        """One scheduling step."""
        if not self.waiting and not self.active:
            # Nothing to do — avoid busy-spinning
            await asyncio.sleep(0.01)
            return

        # Get scheduling decisions
        output: SchedulerOutput = self.strategy.schedule(
            waiting=self.waiting,
            active=self.active,
            memory_manager=self.memory_manager,
            step_id=self._step_id,
        )

        # Move newly admitted requests from waiting to active
        newly_admitted_ids = {
            r.request_id for r in output.prefill_requests
        }
        self.waiting = [
            r for r in self.waiting
            if r.request_id not in newly_admitted_ids
        ]
        for req in output.prefill_requests:
            if req not in self.active:
                self.active.append(req)

        # Run prefill for newly admitted requests
        for req in output.prefill_requests:
            first_token = self.engine.prefill(req)
            req.generated_tokens.append(first_token)

            # Check immediately if first token is EOS
            if self.engine.is_eos(first_token) or len(req.generated_tokens) >= req.max_tokens:
                await self._complete_request(req)
            else:
                # Emit first token to the API stream
                token_text = self.engine.detokenize([first_token])
                await req.output_queue.put(token_text)

        # Run decode step for all decoding requests
        decoding = [
            r for r in output.decode_requests
            if r.status == RequestStatus.DECODING
        ]

        if decoding:
            token_map = self.engine.decode_step(decoding)

            for req in decoding:
                if req.request_id not in token_map:
                    continue

                next_token = token_map[req.request_id]
                req.generated_tokens.append(next_token)

                if self.engine.is_eos(next_token) or len(req.generated_tokens) >= req.max_tokens:
                    await self._complete_request(req)
                else:
                    token_text = self.engine.detokenize([next_token])
                    await req.output_queue.put(token_text)

        self._step_id += 1

    async def _complete_request(self, req: Request) -> None:
        req.status = RequestStatus.COMPLETE
        req.completion_time = time.monotonic()
        self.memory_manager.free(req.request_id)
        self.active = [r for r in self.active if r.request_id != req.request_id]

        # Signal stream completion
        await req.output_queue.put(STREAM_DONE)
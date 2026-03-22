import time
from typing import List
from core.types import Request, RequestStatus, SchedulerOutput
from memory.manager import MemoryManager


class FCFSStrategy:
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size

    def schedule(
        self,
        waiting: List[Request],
        active: List[Request],
        memory_manager: MemoryManager,
        step_id: int,
    ) -> SchedulerOutput:
        prefill_requests = []
        decode_requests = []

        # Split existing active requests into prefill vs decoding
        for req in active:
            if req.status == RequestStatus.PREFILL:
                prefill_requests.append(req)
            elif req.status == RequestStatus.DECODING:
                decode_requests.append(req)

        current_batch_size = len(prefill_requests) + len(decode_requests)

        # Admit waiting requests in arrival order (FCFS)
        # Sort by arrival time to be explicit
        waiting_sorted = sorted(waiting, key=lambda r: r.arrival_time)

        for req in waiting_sorted:
            if current_batch_size >= self.max_batch_size:
                break

            # Check if memory can fit this request's prompt
            if not memory_manager.can_allocate(req.max_tokens):
                # Skip — not enough pages. In FCFS we don't preempt,
                # so we just wait until memory frees up.
                continue

            # Admit the request
            page_ids = memory_manager.allocate(req.request_id, req.max_tokens)
            req.page_ids = page_ids
            req.status = RequestStatus.PREFILL
            req.prefill_start_time = time.monotonic()

            prefill_requests.append(req)
            current_batch_size += 1

        pages_used, pages_total = memory_manager.get_utilization()

        return SchedulerOutput(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            preempted_requests=[],
            step_id=step_id,
            scheduled_at=time.monotonic(),
            available_pages=pages_total - pages_used,
            total_pages=pages_total,
        )
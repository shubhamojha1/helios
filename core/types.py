from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import asyncio

class Priority(Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class Page:
    page_id: int
    is_free: bool = True
    owner_request_id: Optional[str] = None
    token_offset: int = 0
    ref_count: int = 0


class RequestStatus(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    COMPLETE = "complete"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"


@dataclass
class Request:
    request_id: str
    prompt: str
    max_tokens: int
    priority: Priority
    arrival_time: float

    status: RequestStatus = RequestStatus.WAITING
    prompt_tokens: list = field(default_factory=list)
    generated_tokens: list = field(default_factory=list)
    page_ids: list = field(default_factory=list)

    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None

@dataclass
class SchedulerOutput:
    prefill_requests: list
    decode_requests: list
    preempted_requests: list
    step_id: int
    scheduled_at: float
    available_pages: int
    total_pages: int

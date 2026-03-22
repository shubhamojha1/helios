from pydantic import BaseModel
from typing import List, Optional, Literal


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "helios"
    messages: List[Message]
    max_tokens: int = 256
    stream: bool = True
    priority: Literal["high", "normal", "low"] = "normal"


class MetricsResponse(BaseModel):
    pages_used: int
    pages_total: int
    utilization_pct: float
    waiting_requests: int
    active_requests: int
    total_completed: int
import asyncio
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
from core.logger import get_logger

logger = get_logger(__name__)

from api.models import ChatCompletionRequest, MetricsResponse
from core.types import Request, Priority, RequestStatus
from scheduler.scheduler import Scheduler, STREAM_DONE

app = FastAPI(title="Helios Inference Engine")

# Scheduler instance — injected at startup via app.state
# Access via request.app.state.scheduler in endpoints


def _build_prompt(messages) -> str:
    """
    Converts OpenAI-style messages into Mistral instruct format.
    [INST] user message [/INST] assistant message [INST] ...
    """
    prompt = ""
    for msg in messages:
        if msg.role == "user":
            prompt += f"[INST] {msg.content} [/INST]"
        elif msg.role == "assistant":
            prompt += f" {msg.content} "
        elif msg.role == "system":
            # Mistral doesn't have a system token — prepend to first user message
            prompt = f"[INST] {msg.content}\n"
    return prompt.strip()


def _priority_from_str(priority: str) -> Priority:
    return {
        "high": Priority.HIGH,
        "normal": Priority.NORMAL,
        "low": Priority.LOW,
    }[priority]


async def _stream_tokens(
    request: Request,
    scheduler: Scheduler,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted chunks as tokens arrive
    on the request's output_queue.
    """
    while True:
        token = await request.output_queue.get()

        if token is STREAM_DONE:
            # Send final chunk with finish_reason
            final = {
                "id": request.request_id,
                "choices": [{
                    "delta": {"content": ""},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
            break

        chunk = {
            "id": request.request_id,
            "choices": [{
                "delta": {"content": token},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    scheduler: Scheduler = app.state.scheduler
    logger.info(f"Incoming chat completion request: max_tokens={body.max_tokens}, priority={body.priority}")

    prompt = _build_prompt(body.messages)

    req = Request(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        max_tokens=body.max_tokens,
        priority=_priority_from_str(body.priority),
        arrival_time=time.monotonic(),
    )

    await scheduler.submit(req)

    if body.stream:
        return StreamingResponse(
            _stream_tokens(req, scheduler),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Request-ID": req.request_id,
            }
        )
    else:
        # Non-streaming: wait for all tokens then return full response
        tokens = []
        while True:
            token = await req.output_queue.get()
            if token is STREAM_DONE:
                break
            tokens.append(token)

        return {
            "id": req.request_id,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "".join(tokens)
                },
                "finish_reason": "stop"
            }]
        }


@app.post("/v1/cancel/{request_id}")
async def cancel_request(request_id: str):
    logger.info(f"API explicitly cancelled request {request_id}")
    scheduler: Scheduler = app.state.scheduler
    cancelled = await scheduler.cancel(request_id)

    if not cancelled:
        raise HTTPException(status_code=404, detail="Request not found or already complete")

    return {"cancelled": request_id}


@app.get("/v1/metrics", response_model=MetricsResponse)
async def metrics():
    scheduler: Scheduler = app.state.scheduler
    memory = scheduler.memory_manager

    pages_used, pages_total = memory.get_utilization()

    return MetricsResponse(
        pages_used=pages_used,
        pages_total=pages_total,
        utilization_pct=round(pages_used / pages_total * 100, 1),
        waiting_requests=len(scheduler.waiting),
        active_requests=len(scheduler.active),
        total_completed=sum(
            1 for r in scheduler.all_requests.values()
            if r.status == RequestStatus.COMPLETE
        )
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
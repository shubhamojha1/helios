import asyncio
import time
import uuid
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
from core.logger import get_logger

logger = get_logger(__name__)

from api.models import ChatCompletionRequest, CompletionRequest, MetricsResponse
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


def _loaded_model(scheduler: Scheduler) -> str:
    return Path(scheduler.engine.config.model_path).name


def _validate_model(model: str, loaded_model: str) -> None:
    if model == loaded_model:
        return

    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"Model '{model}' not found. Loaded model is '{loaded_model}'.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


async def _stream_tokens(
    request: Request,
    scheduler: Scheduler,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted chunks as tokens arrive
    on the request's output_queue.
    """
    model = Path(scheduler.engine.config.model_path).name
    while True:
        token = await request.output_queue.get()

        if token is STREAM_DONE:
            # Send final chunk with finish_reason
            final = {
                "id": f"chatcmpl-{request.request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "stop",
                }]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
            break

        chunk = {
            "id": f"chatcmpl-{request.request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "logprobs": None,
                "finish_reason": None,
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    scheduler: Scheduler = app.state.scheduler

    loaded_model = _loaded_model(scheduler)
    _validate_model(body.model, loaded_model)
    logger.info(
        f"Incoming chat completion request: model={body.model}, "
        f"max_tokens={body.max_tokens}, priority={body.priority}"
    )

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
            },
        )

    tokens = []
    while True:
        token = await req.output_queue.get()
        if token is STREAM_DONE:
            break
        tokens.append(token)

    return {
        "id": f"chatcmpl-{req.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": loaded_model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(tokens),
            },
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(req.prompt_tokens),
            "completion_tokens": len(req.generated_tokens),
            "total_tokens": len(req.prompt_tokens) + len(req.generated_tokens),
        },
    }


@app.post("/v1/completions")
async def completions(body: CompletionRequest):
    scheduler: Scheduler = app.state.scheduler

    loaded_model = _loaded_model(scheduler)
    _validate_model(body.model, loaded_model)

    log.info("MODEL========> ", loaded_model)

    if body.stream:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Streaming is not implemented for /v1/completions. Use /v1/chat/completions or set stream=false.",
                    "type": "invalid_request_error",
                    "param": "stream",
                    "code": "unsupported_streaming",
                }
            },
        )

    if isinstance(body.prompt, list):
        if len(body.prompt) != 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Only one prompt is supported per request.",
                        "type": "invalid_request_error",
                        "param": "prompt",
                        "code": "unsupported_prompt_batch",
                    }
                },
            )
        prompt = body.prompt[0]
    else:
        prompt = body.prompt

    req = Request(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        max_tokens=body.max_tokens,
        priority=_priority_from_str(body.priority),
        arrival_time=time.monotonic(),
    )

    await scheduler.submit(req)

    tokens = []
    while True:
        token = await req.output_queue.get()
        if token is STREAM_DONE:
            break
        tokens.append(token)

    return {
        "id": f"cmpl-{req.request_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": loaded_model,
        "choices": [{
            "text": "".join(tokens),
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(req.prompt_tokens),
            "completion_tokens": len(req.generated_tokens),
            "total_tokens": len(req.prompt_tokens) + len(req.generated_tokens),
        },
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

@app.get("/v1/models")
async def list_models():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not models_dir.exists():
        return {"object": "list", "data": []}
    
    models = [
        p.name
        for p in models_dir.iterdir()
        if p.is_file() 
        # and p.suffix == ".gguf"
    ]
    return {
        "object": "list",
        "data": [{"id": m, "object": "model"} for m in models],
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

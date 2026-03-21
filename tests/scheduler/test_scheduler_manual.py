import asyncio
import time
import uuid
from core.types import Request, Priority
from engine.engine import Engine, EngineConfig
from memory.manager import MemoryManager
from scheduler.scheduler import Scheduler, STREAM_DONE

# python -m tests.scheduler.test_scheduler_manual

async def main():
    # Setup
    engine = Engine(EngineConfig(
        model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        verbose=False
    ))
    engine.load_model()

    memory = MemoryManager(total_pages=256, page_size_tokens=16)
    scheduler = Scheduler(engine=engine, memory_manager=memory, max_batch_size=2)

    # Submit two requests
    req1 = Request(
        request_id=str(uuid.uuid4()),
        prompt="[INST] What is 2 + 2? [/INST]",
        max_tokens=30,
        priority=Priority.NORMAL,
        arrival_time=time.monotonic(),
    )
    req2 = Request(
        request_id=str(uuid.uuid4()),
        prompt="[INST] Say hello in one word. [/INST]",
        max_tokens=10,
        priority=Priority.NORMAL,
        arrival_time=time.monotonic(),
    )

    await scheduler.submit(req1)
    await scheduler.submit(req2)

    # Run scheduler in background
    scheduler_task = asyncio.create_task(scheduler.run())

    # Consume output from both requests concurrently
    async def consume(req: Request, label: str):
        print(f"\n{label}: ", end="", flush=True)
        while True:
            token = await req.output_queue.get()
            if token is STREAM_DONE:
                break
            print(token, end="", flush=True)
        print(f"\n{label} done.")

    await asyncio.gather(
        consume(req1, "Request 1"),
        consume(req2, "Request 2"),
    )

    scheduler.stop()
    scheduler_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
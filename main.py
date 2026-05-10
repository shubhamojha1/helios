import asyncio
import uvicorn
from api.server import app
from engine.engine import Engine, EngineConfig
from memory.manager import MemoryManager
from core.logger import get_logger
from scheduler.scheduler import Scheduler

logger = get_logger(__name__)

async def main():
    # Initialize components
    engine = Engine(EngineConfig(
        model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    ))
    engine.load_model()

    memory = MemoryManager(total_pages=256, page_size_tokens=16)
    scheduler = Scheduler(engine=engine, memory_manager=memory, max_batch_size=4)

    # Inject scheduler into FastAPI app state
    app.state.scheduler = scheduler

    # Run scheduler and uvicorn server concurrently
    config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="warning")
    logger.info(f"Host: {config.host} | PORT: {config.port}")
    server = uvicorn.Server(config)

    await asyncio.gather(
        scheduler.run(),
        server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
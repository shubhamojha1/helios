import asyncio
import uvicorn
from api.server import app
from engine.engine import Engine, EngineConfig
from memory.manager import MemoryManager
from core.logger import get_logger
from scheduler.scheduler import Scheduler
from engine.engine_torch import TorchEngine

logger = get_logger(__name__)


async def serve(
    model_path: str = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    host: str = "127.0.0.1",
    port: int = 8080,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
) -> None:
    # Initialize components
    # engine = Engine(EngineConfig(
    #     model_path=model_path,
    #     n_gpu_layers=n_gpu_layers,
    #     n_ctx=n_ctx,
    #     verbose=False,
    # ))
    engine = TorchEngine(EngineConfig(
        model_path="models/qwen2.5-3b-instruct"
    ))
    engine.load_model()

    memory = MemoryManager(total_pages=256, page_size_tokens=16)
    scheduler = Scheduler(engine=engine, memory_manager=memory)

    # Inject scheduler into FastAPI app state
    app.state.scheduler = scheduler

    # Run scheduler and uvicorn server concurrently
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    logger.info(f"Host: {config.host} | PORT: {config.port}")
    server = uvicorn.Server(config)

    await asyncio.gather(
        scheduler.run(),
        server.serve(),
    )


async def main():
    await serve()


if __name__ == "__main__":
    asyncio.run(main())
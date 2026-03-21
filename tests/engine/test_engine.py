from engine.engine import Engine, EngineConfig
from core.types import Request, Priority, RequestStatus
import time
import uuid
import pytest

def test_engine_loads():
    config = EngineConfig(
        model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        verbose=False
    )
    engine = Engine(config)
    engine.load_model()
    assert engine.llm is not None
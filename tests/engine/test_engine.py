from engine.engine import Engine, EngineConfig
from core.types import Request, Priority, RequestStatus
import time
import uuid

# From project root:
# python -m pytest tests/memory/test_memory.py -v


def make_request(prompt: str) -> Request:
    return Request(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        max_tokens=50,
        priority=Priority.NORMAL,
        arrival_time=time.monotonic(),
    )

def test_engine_loads():
    config = EngineConfig(
        model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        verbose=False
    )
    engine = Engine(config)
    engine.load_model()
    assert engine.llm is not None

def test_tokenize_detokenize():
    config = EngineConfig(model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    engine = Engine(config)
    engine.load_model()

    text = "Hello, world!"
    tokens = engine.tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    recovered = engine.detokenize(tokens)
    assert "Hello" in recovered

def test_prefill_and_decode():
    config = EngineConfig(model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    engine = Engine(config)
    engine.load_model()

    req = make_request("[INST] Say hello. [/INST]")
    req.prompt_tokens = engine.tokenize(req.prompt)

    # Prefill — should return first token and transition status
    first_token = engine.prefill(req)
    assert isinstance(first_token, int)
    assert req.status == RequestStatus.DECODING
    assert req.first_token_time is not None

    # Simulate decode steps
    req.generated_tokens = [first_token]
    generated = [first_token]

    for _ in range(20):
        results = engine.decode_step([req])
        assert req.request_id in results

        next_token = results[req.request_id]
        req.generated_tokens.append(next_token)
        generated.append(next_token)

        if engine.is_eos(next_token):
            break

    output = engine.detokenize(generated)
    print(f"\nGenerated: {output}")
    assert len(output) > 0


def test_is_eos():
    config = EngineConfig(model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    engine = Engine(config)
    engine.load_model()

    # EOS token should not be a regular word token
    assert isinstance(engine.is_eos(2), bool)
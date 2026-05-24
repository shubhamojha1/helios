# test_torch_engine.py
from engine.engine_torch import TorchEngine
from core.types import Request, Priority
import uuid, time

engine = TorchEngine("models/qwen2.5-3b-instruct")
engine.load_model()

req = Request(
    request_id=str(uuid.uuid4()),
    prompt="What is the capital of France?",
    max_tokens=50,
    priority=Priority.LOW,
    arrival_time=time.monotonic()
)
req.prompt_tokens = engine.tokenize(req.prompt)

first_token = engine.prefill(req)
req.generated_tokens.append(first_token)
print("First token:", engine.detokenize([first_token]))

for _ in range(10):
    token_map = engine.decode_step([req])
    token = token_map[req.request_id]
    req.generated_tokens.append(token)
    print(engine.detokenize([token]), end="", flush=True)
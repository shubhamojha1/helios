from dataclasses import dataclass
from typing import Dict, List, Optional
from llama_cpp import Llama
from core.types import Request, RequestStatus
import time


@dataclass
class EngineConfig:
    model_path: str
    n_ctx: int = 2048
    n_gpu_layers: int = -1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    verbose: bool = False

class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.llm : Optional[Llama] = None

    def load_model(self) -> None:
        print(f"Loading model from {self.config.model_path}...")
        self.llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=self.config.verbose,
        )
        print("Model loaded!")

    def tokenize(self, text: str) -> List[int]:
        assert self.llm is not None, "Model not loaded"
        return self.llm.tokenize(text.encode("utf-8"))

    def detokenize(self, token_ids: List[int]) -> str:
        assert self.llm is not None, "Model not loaded"
        return self.llm.detokenize(token_ids).decode("utf-8", errors="ignore")

    def prefill(self, request: Request) -> int:
        """
        Runs the prompt through the model (prefill pass).
        Populates the KV Cache for all prompt tokens.
        Returns the first generated token.

        This is the expensive step. Cost scales with prompt length.
        """ 
        assert self.llm is not None, "Model not loaded"

        request.prefill_start_time = time.monotonic()
        request.status = RequestStatus.PREFILL

        # Feed the full prompt into the model
        self.llm.reset() # Clear any previous KV cache state
        self.llm.eval(request.prompt_tokens)

        first_token = self.llm.sample(
            temp=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
        )

        request.first_token_time = time.monotonic()
        request.status = RequestStatus.DECODING
        return first_token

    def decode_step(self, requests: List[Request]) -> Dict[str, int]:
        """
        Runs one decoding step for each request sequentially.
        Returns a mapping of request_id -> next_token_id

        NOTE: This runs each request's forward pass sequentially, not
        in  a true parallel GPU batch. True batching would require direct
        control over CUDA kernels and KV cache memory layout, which
        llama-cpp-python does not expose. The scheduling behavior is 
        identical to true continuous batching. Only absolute throughput
        numbers differ.
        """
        assert self.llm is not None, "Model not loaded"

        results: Dict[str, int] = {}

        for request in requests:
            if not request.generated_tokens:
                # Should not happen. Prefill always produces the first token
                continue

            last_token = request.generated_tokens[-1]
            self.llm.eval([last_token])

            next_token = self.llm.sample(
                temp=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
            )

            results[request.request_id] = next_token

        return results

    def is_eos(self, token_id: int) -> bool:
        assert self.llm is not None, "Model not loaded"
        return token_id == self.llm.token_eos()
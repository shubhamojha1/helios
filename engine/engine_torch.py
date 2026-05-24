import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer
from core.types import Request, RequestStatus

from core.logger import get_logger

logger = get_logger(__name__)

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

class TorchEngine:
    # def __init__(self, model_path: str):
    def __init__(self, config: EngineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        # KV Cache per request: request_id -> (past_key_values)
        self.kv_cache: Dict[str, Tuple] = {}

    def load_model(self) -> None:
        logger.info(f"Loading model from {self.config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, dtype=torch.bfloat16, device_map="cuda"
        )
        self.model.eval()
        logger.info("Model loaded.")

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def prefill(self, request: Request) -> int:
        request.status = RequestStatus.PREFILL
        request.prefill_start_time = time.monotonic()

        input_ids = torch.tensor(
            [request.prompt_tokens],
            dtype=torch.long,
            device="cuda"
        )

        with torch.no_grad():
            output = self.model(
                input_ids = input_ids,
                use_cache = True,
                return_dict = True
            )

        # Store KV Cache for this request
        self.kv_cache[request.request_id] = output.past_key_values

        # Sample first token
        first_token = self._sample(output.logits[0, -1, :])

        request.first_token_time = time.monotonic()
        request.status = RequestStatus.DECODING
        return first_token

    @staticmethod
    def _make_cache(layer_tensors: list) -> DynamicCache:
        """Build a DynamicCache from a list of (key, value) tensor pairs, one per layer."""
        cache = DynamicCache()
        for k, v in layer_tensors:
            layer = DynamicLayer()
            layer.keys = k
            layer.values = v
            layer.dtype = k.dtype
            layer.device = k.device
            layer.is_initialized = True
            cache.layers.append(layer)
        return cache

    def decode_step(self, requests: List[Request]) -> Dict[str, int]:
        requests = [r for r in requests if r.request_id in self.kv_cache]
        if not requests:
            return {}

        # last generated token per request → (B, 1)
        input_ids = torch.tensor(
            [[r.generated_tokens[-1]] for r in requests],
            dtype=torch.long,
            device="cuda",
        )

        # transformers 5.x DynamicCache: layers[i].keys / .values
        caches = [self.kv_cache[r.request_id] for r in requests]
        past_seq_lens = [c.layers[0].keys.shape[2] for c in caches]
        max_past_len = max(past_seq_lens)
        n_layers = len(caches[0].layers)

        # left-pad each request's KV cache to max_past_len, then stack into batch
        batched_layers = []
        for layer_idx in range(n_layers):
            keys, values = [], []
            for cache, past_len in zip(caches, past_seq_lens):
                k = cache.layers[layer_idx].keys    # (1, heads, past_len, head_dim)
                v = cache.layers[layer_idx].values
                pad = max_past_len - past_len
                keys.append(F.pad(k, (0, 0, pad, 0)))    # pad start of seq dim
                values.append(F.pad(v, (0, 0, pad, 0)))
            batched_layers.append((
                torch.cat(keys, dim=0),     # (B, heads, max_past_len, head_dim)
                torch.cat(values, dim=0),
            ))
        batched_cache = self._make_cache(batched_layers)

        # attention mask: 1 for real tokens, 0 for left-padding
        attention_mask = torch.zeros(
            len(requests), max_past_len + 1, dtype=torch.long, device="cuda"
        )
        for i, past_len in enumerate(past_seq_lens):
            attention_mask[i, max_past_len - past_len:] = 1

        # position_ids: each request's true next-token position (critical for RoPE)
        position_ids = torch.tensor(
            [[past_len] for past_len in past_seq_lens],
            dtype=torch.long,
            device="cuda",
        )

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                past_key_values=batched_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )

        # unpad the updated cache and store back per-request, then sample
        updated = output.past_key_values
        results = {}
        for i, (r, past_len) in enumerate(zip(requests, past_seq_lens)):
            start = max_past_len - past_len
            self.kv_cache[r.request_id] = self._make_cache([
                (updated.layers[layer_idx].keys[i:i+1, :, start:, :],
                 updated.layers[layer_idx].values[i:i+1, :, start:, :])
                for layer_idx in range(n_layers)
            ])
            results[r.request_id] = self._sample(output.logits[i, -1, :])

        return results

    def _sample(self, logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        # top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
        sorted_probs[sorted_indices_to_remove] = 0
        sorted_probs /= sorted_probs.sum()

        next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
        return next_token.item()

    def is_eos(self, token_id: int) -> bool:
        return token_id == self.tokenizer.eos_token_id

    def free(self, request_id: str) -> None:
        self.kv_cache.pop(request_id, None)

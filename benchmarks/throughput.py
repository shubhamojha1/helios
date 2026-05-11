import time
import asyncio
from openai import AsyncOpenAI
# python -m benchmarks.throughput
client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="helios")

async def single_request(prompt: str, max_tokens: int) -> dict: 
    start = time.perf_counter()
    first_token_at = None

    stream = await client.chat.completions.create(
        model="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True
    )

    completion_tokens = 0
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            completion_tokens += 1
    
    end = time.perf_counter()
    ttft_ms = (first_token_at - start) * 1000 if first_token_at is not None else float("nan")
    return {
        "ttft_ms": ttft_ms,
        "total_ms": (end - start) * 1000,
        "tokens": completion_tokens,
        "tokens_per_sec": completion_tokens / (end - start)
    }

async def bench(concurrency: int, max_tokens: int = 256):
    await single_request("warmup", max_tokens=10)

    prompt = "Explain the architecture of a transformer model in detail."
    tasks = [single_request(prompt, max_tokens) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)

    ttfts = [r["ttft_ms"] for r in results]
    tps = [r["tokens_per_sec"] for r in results]
    print(f"\nconcurrency={concurrency}")
    print(f"  TTFT p50={sorted(ttfts)[len(ttfts)//2]:.1f}ms  p95={sorted(ttfts)[int(len(ttfts)*.95)]:.1f}ms")
    print(f"  Throughput: {sum(tps):.1f} tokens/sec total")


if __name__ == "__main__":
    
    for c in [1, 4, 8, 16]:
        asyncio.run(bench(concurrency=c))

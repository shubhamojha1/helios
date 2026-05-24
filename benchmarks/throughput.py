import time
import asyncio
import json
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

# python -m benchmarks.throughput
client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="helios")

OUTPUT_DIR = Path(__file__).parent / "output"

async def single_request(prompt: str, max_tokens: int) -> dict: 
    start = time.perf_counter()
    first_token_at = None

    stream = await client.chat.completions.create(
        model="qwen2.5-3b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
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

async def bench(concurrency: int, max_tokens: int = 256) -> dict:
    await single_request("warmup", max_tokens=10)

    prompt = "Explain the architecture of a transformer model in detail."
    tasks = [single_request(prompt, max_tokens) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)

    ttfts = sorted(r["ttft_ms"] for r in results)
    tps = [r["tokens_per_sec"] for r in results]
    p50 = ttfts[len(ttfts) // 2]
    p95 = ttfts[min(int(len(ttfts) * 0.95), len(ttfts) - 1)]

    print(f"\nconcurrency={concurrency}")
    print(f"  TTFT p50={p50:.1f}ms  p95={p95:.1f}ms")
    print(f"  Throughput: {sum(tps):.1f} tokens/sec total")

    return {
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "ttft_p50_ms": round(p50, 2),
        "ttft_p95_ms": round(p95, 2),
        "throughput_tokens_per_sec": round(sum(tps), 2),
        "per_request": results,
    }


if __name__ == "__main__":
    run_results = []
    for c in [1, 4, 8, 16]:
        result = asyncio.run(bench(concurrency=c))
        if result:
            run_results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"throughput_{timestamp}.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(run_results, indent=2))
    print(f"\nResults saved to {out_path}")

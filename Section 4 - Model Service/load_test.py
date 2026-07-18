import asyncio
import time
from typing import Dict
import httpx

API_URL = "http://localhost:8000/generate"
CONCURRENT_REQUESTS = 10
PROMPT = "Write a short paragraph explaining the theory of relativity."


async def send_request(client: httpx.AsyncClient, req_id: int) -> Dict:
    payload = {
        "prompt": PROMPT,
        "max_new_tokens": 128,
        "temperature": 0.7,
    }

    start_time = time.perf_counter()
    ttft = None
    total_time = None
    chunks_count = 0
    full_text = ""

    try:
        async with client.stream(
            "POST", API_URL, json=payload, timeout=60.0
        ) as response:
            if response.status_code != 200:
                print(f"Request {req_id} failed with status {response.status_code}")
                return {
                    "id": req_id,
                    "success": False,
                    "error": f"Status {response.status_code}",
                }

            async for chunk in response.aiter_text():
                if not chunk:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                full_text += chunk
                chunks_count += 1

        total_time = time.perf_counter() - start_time
        return {
            "id": req_id,
            "success": True,
            "ttft": ttft if ttft is not None else 0.0,
            "total_latency": total_time,
            "tokens_generated": chunks_count,  # Approx token count based on streaming chunks
            "response_length": len(full_text),
        }
    except Exception as e:
        print(f"Request {req_id} failed with exception: {e}")
        return {"id": req_id, "success": False, "error": str(e)}


async def run_load_test():
    print(f"Starting load test with {CONCURRENT_REQUESTS} concurrent requests...")
    print(f"Target URL: {API_URL}")
    print(f"Prompt: {PROMPT}\n")

    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i) for i in range(1, CONCURRENT_REQUESTS + 1)]
        results = await asyncio.gather(*tasks)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("=" * 55)
    print("  LOAD TEST RESULTS")
    print("=" * 55)
    print(f"Successful Requests: {len(successful)} / {CONCURRENT_REQUESTS}")
    print(f"Failed Requests:     {len(failed)} / {CONCURRENT_REQUESTS}\n")

    if successful:
        avg_ttft = sum(r["ttft"] for r in successful) / len(successful)
        avg_latency = sum(r["total_latency"] for r in successful) / len(successful)
        avg_tokens = sum(r["tokens_generated"] for r in successful) / len(successful)

        print(f"Average Time-to-First-Token (TTFT): {avg_ttft:.3f}s")
        print(f"Average Total Latency:              {avg_latency:.3f}s")
        print(f"Average Tokens Generated:            {avg_tokens:.1f}")

        print("\nIndividual Latencies:")
        print(
            f"{'Request ID':<12}{'TTFT (s)':<12}{'Total Latency (s)':<20}{'Tokens':<10}"
        )
        print("-" * 55)
        for r in successful:
            print(
                f"{r['id']:<12}{r['ttft']:<12.3f}{r['total_latency']:<20.3f}{r['tokens_generated']:<10}"
            )
    else:
        print("No successful requests to report stats on.")


if __name__ == "__main__":
    asyncio.run(run_load_test())

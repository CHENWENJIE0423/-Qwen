import asyncio
import aiohttp
import json
import time

API_URL = "http://0.0.0.0:8005/generate"
REQUEST_BODY = {
    "query": "写一段200字的作文"
}

latencies = []
successful_requests = 0
total_requests = 0
total_tokens = 0


async def fetch(session, semaphore, idx):
    global successful_requests, total_requests, total_tokens
    async with semaphore:
        start_time = time.time()  # 请求开始时间
        try:
            async with session.post(API_URL, json=REQUEST_BODY) as response:
                latency = (time.time() - start_time) * 1000  # 转换为毫秒
                latencies.append(latency)
                total_requests += 1

                if response.status == 200:
                    successful_requests += 1
                    full_response = ""

                    async for line in response.content:
                        data = json.loads(line.decode("utf-8"))
                        full_response += data["response"]
                        token_count = len(data["response"])
                        total_tokens += token_count

                    result = {
                        "response": full_response,
                        "status": response.status
                    }
                    print(f"Response {idx + 1}: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    return result
                else:
                    print(f"Response {idx + 1} Error: {response.status}")
                    return None
        except Exception as e:
            print(f"Response {idx + 1} Exception: {str(e)}")
            return None


async def benchmark(concurrency, num_requests):
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, semaphore, i) for i in range(num_requests)]

        start_time = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        rps = num_requests / total_time if total_time > 0 else 0
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        print(f"\nPerformance Metrics with Concurrency {concurrency}:")
        print(f"  ○ Token吞吐率: {tokens_per_second:.2f} tokens/s")
        print(f"  ○ 平均响应时间 (ms): {avg_latency:.2f} ms")
        print(f"  ○ 每秒请求数 (RPS): {rps:.2f} requests/s")
        print(f"  ○ 总耗时 (s): {total_time:.2f} seconds")


if __name__ == '__main__':
    concurrency_levels = [1, 2, 5]
    num_requests = 10

    for concurrency in concurrency_levels:
        print(f"\nRunning benchmark with concurrency level: {concurrency}")
        asyncio.run(benchmark(concurrency, num_requests))
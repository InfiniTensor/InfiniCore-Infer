import asyncio
import time
from openai import AsyncOpenAI
import argparse
import random


PROMPTS = [
    "如果猫能写诗，它们会写些什么？",
    "描述一个没有重力的世界。",
    "如果地球停止自转，会发生什么？",
    "假设你是一只会飞的鲸鱼，描述你的日常生活。",
    "如果人类可以与植物沟通，世界会变成什么样？",
    "描述一个由糖果构成的城市。",
    "如果时间旅行成为可能，你最想去哪个时代？",
    "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
    "如果动物能上网，它们会浏览什么网站？",
    "描述一个没有声音的世界。",
    "如果人类可以在水下呼吸，城市会如何变化？",
    "想象一下，如果天空是绿色的，云是紫色的。",
    "如果你能与任何历史人物共进晚餐，你会选择谁？",
    "描述一个没有夜晚的星球。",
    "如果地球上只有一种语言，世界会如何运作？",
    "想象一下，如果所有的书都变成了音乐。",
    "如果你可以变成任何一种动物，你会选择什么？",
    "描述一个由机器人统治的未来世界。",
    "如果你能与任何虚构角色成为朋友，你会选择谁？",
    "想象一下，如果每个人都能读懂他人的思想。"
]

NUM_REQUESTS = 10
CONCURRENCY = 5
API_URL = "http://127.0.0.1:8010"
MODEL = "FM9G-7B"


async def benchmark_user(client, semaphore, queue, results, user_id, verbose):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break

            question = random.choice(PROMPTS)
            try: 
                print(f"🚀 User#{user_id} Sending request #{task_id}")

                start_time = time.time()
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": question}],
                    stream=True
                )

                first_token_time = None
                total_tokens = 0
                answer_chunks = []

                async for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    delta = chunk.choices[0].delta.content
                    if delta:
                        answer_chunks.append(delta)
                        total_tokens += 1
                    if chunk.choices[0].finish_reason is not None:
                        break

                end_time = time.time()

                ttft = first_token_time - start_time if first_token_time else None
                elapsed_time = end_time - start_time if start_time else None
                ms_per_token = (elapsed_time / total_tokens * 1000) if total_tokens > 0 and elapsed_time else None
                tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

                answer = "".join(answer_chunks)

                results.append((total_tokens, elapsed_time, tokens_per_second, ttft, ms_per_token))

                if verbose:
                    print(f"\n📝 Request #{task_id} (User #{user_id})")
                    print(f"  ⏱ 首字延迟 TTFT: {ttft:.3f}s")
                    print(f"  ⏱ 总耗时: {elapsed_time:.3f}s")
                    print(f"  🔤 解码 token 总数: {total_tokens}")
                    print(f"  📏 平均 token 解码时间: {ms_per_token:.2f} ms/token")
                    print(f"  ❓ 提问: {question}")
                    print(f"  💬 回答: {answer}\n")

                queue.task_done()
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Request #{task_id} (User #{user_id}) FAILED:")
                    print(f"  ❌ Error: {e}\n")

async def run_benchmark(verbose=False):
    client = AsyncOpenAI(base_url=API_URL, api_key="default")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    results = []
    for i in range(NUM_REQUESTS):
        await queue.put(i)
    for _ in range(CONCURRENCY):
        await queue.put(None)

    users = [
        asyncio.create_task(benchmark_user(client, semaphore, queue, results, user_id, verbose))
        for user_id in range(CONCURRENCY)
    ]

    start_time = time.time()
    await queue.join()
    await asyncio.gather(*users)
    end_time = time.time()

    total_elapsed_time = end_time - start_time
    tokens_list = [r[0] for r in results if r and r[0] is not None]
    latencies = [r[1] for r in results if r and r[1] is not None]
    tokens_per_second_list = [r[2] for r in results if r and r[2] is not None]
    ttft_list = [r[3] for r in results if r and r[3] is not None]
    ms_per_token_list = [r[4] for r in results if r and r[4] is not None]

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_ms_per_token = sum(ms_per_token_list) / len(ms_per_token_list) if ms_per_token_list else None

    width_label = 24
    sep = "-" * 60

    print(f"\n=== 📊 性能指标汇总 ({MODEL}) ===")
    print(sep)
    print(f"{'并发数':<{width_label}}: {CONCURRENCY}")
    print(f"{'请求总数':<{width_label}}: {NUM_REQUESTS}")
    print(f"{'成功请求数':<{width_label}}: {successful_requests}")
    print(f"{'总耗时':<{width_label}}: {total_elapsed_time:.2f} s")
    print(f"{'总输出token数':<{width_label}}: {sum(tokens_list)}")
    print(f"{'请求速率 (RPS)':<{width_label}}: {requests_per_second:.2f} requests/s")
    print(sep)
    print(f"{'Average latency':<{width_label}}: {avg_latency:.2f} s")
    print(f"{'Average TTFT':<{width_label}}: {avg_ttft:.2f} s")
    print(f"{'Avg time per token':<{width_label}}: {avg_ms_per_token:.2f} ms/token")
    print(f"{'Avg Token generation speed':<{width_label}}: {avg_tokens_per_second:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        args.verbose
    ))

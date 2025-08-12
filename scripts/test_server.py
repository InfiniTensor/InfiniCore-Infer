import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8010/chat/completions"
MODEL = "FM9G-7B"
PROMPT = ["山东最高的山是？", "给我讲个故事"]
CONCURRENCY = 10  # 并发用户数量

def single_run(user_id):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT[user_id % len(PROMPT)]}],
        "max_tokens": 512,
        "stream": True
    }
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    print(f"[User {user_id}] Sending request...")
    
    start = time.perf_counter()
    resp = requests.post(API_URL, headers=headers, json=payload, stream=True)
    resp.raise_for_status()
    
    ttfb = resp.elapsed.total_seconds()  # HTTP header 到达时间
    header_received = time.perf_counter()
    
    if resp.encoding is None:
        resp.encoding = 'utf-8'
    
    tokens = 0
    chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or line.strip() == "[DONE]":
            continue
        s = line.strip()
        if s.startswith("data:"):
            s = s[len("data:"):].strip()
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            continue
        text = data.get("choices", [{}])[0].get("delta", {}).get("content")
        if text:
            chunks.append(text)
            tokens += 1
    stream_done = time.perf_counter()
    
    # 时间计算
    stream_time = stream_done - header_received
    total_time = stream_done - start
    time_per_token_ms = (stream_time / tokens * 1000) if tokens else float('inf')
    tps = tokens / stream_time if stream_time > 0 else 0
    
    
    return {
        "user": user_id,
        "ttfb": ttfb,
        "stream_time": stream_time,
        "total_time": total_time,
        "tokens": tokens,
        "time_per_token_ms": time_per_token_ms,
        "tps": tps,
        "chunks": chunks
    }

def main():
    worst = None 
    worst_stream = -1.0
    best_stream = float('inf')
    results = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as e:
        futures = [e.submit(single_run, uid) for uid in range(CONCURRENCY)]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)

            print(
                f"User {r['user']} → TTFB = {r['ttfb']:.3f}s, latency = {r['stream_time']:.3f}s, "
                f"tokens = {r['tokens']}, time/token = {r['time_per_token_ms']:.2f} ms, "
                f"TPS = {r['tps']:.1f} tok/s"
            )
            if r['stream_time'] > worst_stream:
                worst_stream = r['stream_time']
                worst = r
            if r['stream_time'] < best_stream:
                best_stream = r['stream_time']
                best = r
    
    # Sort results by user ID
    results.sort(key=lambda x: x["user"])

    with open("responses.txt", "w", encoding="utf-8") as fw:
        for r in results:
            fw.write(f"[User {r['user']}]\n")
            text = "".join(r["chunks"])
            # fixed = text.encode('latin-1').decode('utf-8')
            fixed = text
            fw.write(fixed)
            fw.write("\n\n")

    n = CONCURRENCY
    avg_ttfb = sum(r['ttfb'] for r in results) / n
    avg_token = sum(r['tokens'] for r in results) / n
    avg_stream = sum(r['stream_time'] for r in results) / n
    avg_tps = sum(r['tps'] for r in results) / n
    avg_time_per_token = sum(r['time_per_token_ms'] for r in results) / n

    print(f"\n✅ All {n} requests completed.")
    print(f"Averages → TTFB = {avg_ttfb:.3f}s, latency = {avg_stream:.3f}s, "
          f"tokens = {avg_token:.1f}, TPS = {avg_tps:.1f} tok/s, time/token = {avg_time_per_token:.2f} ms")

    if best:
        print("\nFastest user:")
        print(
            f"User {best['user']} → latency = {best['stream_time']:.3f}s, "
            f"tokens = {best['tokens']}, TPS = {best['tps']:.1f} tok/s, "
            f"time/token = {best['time_per_token_ms']:.2f} ms"
        )
    if worst:
        print("\nSlowest user:")
        print(
            f"User {worst['user']} → latency = {worst['stream_time']:.3f}s, "
            f"tokens = {worst['tokens']}, TPS = {worst['tps']:.1f} tok/s, "
            f"time/token = {worst['time_per_token_ms']:.2f} ms"
        )

if __name__ == "__main__":
    main()

from jiuge import JiugeForCauslLM
from libinfinicore_infer import DeviceType
from infer_task import InferTask
from kvcache_pool import KVCachePool
from dynamic_batch_manager import DynamicBatchManager

import argparse
import queue
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import contextlib
import uvicorn
import time
import uuid
import json
import threading
import janus
import torch
import pynvml
import psutil


DEVICE_TYPE_MAP = {
    "cpu": DeviceType.DEVICE_TYPE_CPU,
    "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
    "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
    "ascend": DeviceType.DEVICE_TYPE_ASCEND,
    "metax": DeviceType.DEVICE_TYPE_METAX,
    "moore": DeviceType.DEVICE_TYPE_MOORE,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Launch the LLM inference server.")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--dev",
        type=str,
        choices=DEVICE_TYPE_MAP.keys(),
        default="cpu",
        help="Device type to run the model on (default: cpu)",
    )
    parser.add_argument(
        "--ndev",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=5,
        help="Maximum number of requests that can be batched together (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=False,
        default=None,
        help="Max token sequence length that model will handle (follows model config if not provided)",
    )
    return parser.parse_args()

args = parse_args()
device_type = DEVICE_TYPE_MAP[args.dev]
model_path = args.model_path
ndev = args.ndev
max_tokens = args.max_tokens

MAX_BATCH = args.max_batch
print(
    f"Using MAX_BATCH={MAX_BATCH}. Try reduce this value if out of memory error occurs."
)

def chunk_json(id_, content=None, role=None, finish_reason=None):
    delta = {}
    if content:
        delta["content"] = content
    if role:
        delta["role"] = role
    return {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "jiuge",
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


# A wrapper for InferTask that supports async output queue
class AsyncInferTask(InferTask):
    def __init__(self, id, tokens, max_tokens, temperature, topk, topp, end_tokens):
        super().__init__(id, tokens, max_tokens, temperature, topk, topp, end_tokens)
        self.output_queue = janus.Queue()
        print(f"[INFO] Create InferTask {self.id}")

    def output(self, out_token):
        self.next(out_token)
        self.output_queue.sync_q.put(out_token)

def get_memory_usage() -> float:
    """获取当前GPU显存使用率，如果GPU不可用则获取系统内存使用率"""
    try:
        # 检查是否有可用的GPU
        if pynvml and hasattr(pynvml, 'nvmlInit'):
            try:
                pynvml.nvmlInit()
                # 使用第一个GPU设备
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # 清理PyTorch缓存以获得准确的显存使用率
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 获取GPU显存使用率
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                gpu_usage = memory_info.used / memory_info.total
                return gpu_usage
            except Exception as e:
                print(f"[WARNING] Failed to get GPU memory usage: {e}")
        
        # 回退到系统内存使用率
        memory = psutil.virtual_memory()
        return memory.percent / 100.0
        
    except Exception as e:
        print(f"[WARNING] Failed to get memory usage: {e}")
        return 0.5  # 默认返回50%


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.model = JiugeForCauslLM(model_path, device_type, ndev, max_tokens=max_tokens)
    app.state.kv_cache_pool = KVCachePool(app.state.model, MAX_BATCH)
    app.state.request_queue = janus.Queue()
    initial_mem_usage = get_memory_usage()
    print(f'[Info] initial memory usage: {initial_mem_usage}')
    # 初始化动态批处理管理器
    app.state.batch_manager = DynamicBatchManager(
        min_batch_size=1,
        max_batch_size=MAX_BATCH,
        max_wait_time_ms=200,  # 增加等待时间以允许更大的批次
        memory_threshold=0.9,  # 调整内存阈值到90%，避免过早触发内存压力保护
        base_mem_usage=initial_mem_usage,
        gpu_device_id=0  # 使用第一个GPU设备
    )
    worker_thread = threading.Thread(target=worker_loop, args=(app,), daemon=True)
    worker_thread.start()

    try:
        yield  # The app runs here
    finally:
        # Shutdown
        app.state.request_queue.sync_q.put(None)
        worker_thread.join()
        app.state.request_queue.shutdown()

        app.state.kv_cache_pool.finalize()
        app.state.model.destroy_model_instance()


App = FastAPI(lifespan=lifespan)


@App.get("/batch_stats")
async def get_batch_stats():
    """获取动态批处理统计信息"""
    try:
        stats = App.state.batch_manager.get_stats()
        return JSONResponse(content={
            "status": "success",
            "data": stats
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# App loop: take requests from the queue, do inference, and put unfinished requests back into the queue.
def worker_loop(app):
    """动态批处理工作循环"""
    batch_manager = app.state.batch_manager
    
    while True:
        try:
            # 尝试获取第一个任务
            task = app.state.request_queue.sync_q.get(timeout=0.05)
        except queue.Empty:
            continue

        if task is None:
            return

        # 开始构建批次
        batch = [task]
        batch_start_time = time.time()
        
        while True:
            
            queue_size = app.state.request_queue.sync_q.qsize()
            
            # 获取动态批处理建议
            suggested_batch_size = batch_manager.calculate_dynamic_batch_size(
                queue_size + len(batch)
            )

            current_time = time.time()
            wait_time_ms = (current_time - batch_start_time) * 1000
            # 判断是否应该处理当前批次
            should_process = batch_manager.should_process_batch(
                len(batch),suggested_batch_size, queue_size, wait_time_ms
            )
            
            if should_process:
                break
            
            # 尝试添加更多任务到批次
            if len(batch) < suggested_batch_size:
                try:
                    # 计算剩余等待时间，确保有足够时间收集更多请求
                    remaining_wait_ms = max(0, batch_manager.current_wait_time_ms - wait_time_ms)
                    timeout_seconds = min(0.05, remaining_wait_ms / 1000.0)  # 最多等待50ms
                    req = app.state.request_queue.sync_q.get(timeout=timeout_seconds)
                    if req is not None:
                        batch.append(req)
                    else:
                        break
                except queue.Empty:
                    # 检查是否应该继续等待
                    if wait_time_ms >= batch_manager.current_wait_time_ms:
                        break
                    time.sleep(0.001)  # 短暂等待
            else:
                break
        
        # 执行批量推理
        if len(batch) > 0:
            print(f"[DEBUG] Processing batch with size: {len(batch)}, suggested size was: {suggested_batch_size}")
            infer_start_time = time.time()
            
            try:
                output_tokens = app.state.model.batch_infer_one_round(batch)
                
                # 处理输出
                finished_tasks = 0
                for task, token in zip(batch, output_tokens):
                    task.output(token)
                    if task.finish_reason is None:
                        app.state.request_queue.sync_q.put(task)
                    else:
                        print(f"[INFO] Task {task.id} finished infer.")
                        app.state.kv_cache_pool.release_sync(task)
                        finished_tasks += 1
                
                # 如果有任务完成，检查是否需要清理显存
                if finished_tasks > 0:
                    current_memory = batch_manager.get_memory_usage()
                    if current_memory > 0.75:  # 显存占用超过75%时主动清理
                        print(f"[INFO] Memory usage before cleanup: {current_memory:.2%}")
                        batch_manager._force_memory_cleanup()
                        # 清理后再次检查显存使用率
                        new_memory = batch_manager.get_memory_usage()
                        print(f"[INFO] Memory usage after cleanup: {new_memory:.2%}, freed: {(current_memory-new_memory)*100:.2f}%")
                    elif finished_tasks >= 3:  # 每完成3个任务就强制清理一次
                        print(f"[INFO] Periodic cleanup after {finished_tasks} tasks, memory usage: {current_memory:.2%}")
                        batch_manager._force_memory_cleanup()
                
                # 记录性能数据
                infer_end_time = time.time()
                latency_ms = (infer_end_time - infer_start_time) * 1000
                throughput = len(batch) / (infer_end_time - infer_start_time)
                
                batch_manager.record_batch_performance(
                    len(batch), latency_ms, throughput
                )
                
                # 定期打印统计信息
                if len(batch_manager.batch_history) % 50 == 0:
                    stats = batch_manager.get_stats()
                    print(f"[INFO] Dynamic Batch Stats: optimal_batch={stats['current_optimal_batch']}, "
                          f"wait_time={stats['current_wait_time_ms']}ms, "
                          f"memory_usage={stats['memory_usage']:.4%}, "
                          f"avg_latency={stats['avg_latency']:.1f}ms, "
                          f"avg_throughput={stats['avg_throughput']:.1f} req/s")
                
            except Exception as e:
                print(f"[ERROR] Batch inference failed: {e}")
                # 将任务重新放回队列
                for task in batch:
                    if task.finish_reason is None:
                        app.state.request_queue.sync_q.put(task)


def build_task(id_, request_data, request: Request):
    messages = request_data.get("messages", [])
    input_content = request.app.state.model.tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    tokens = request.app.state.model.tokenizer.encode(input_content)
    return AsyncInferTask(
        id_,
        tokens,
        request_data.get("max_tokens", request.app.state.model.max_context_len()),
        request_data.get("temperature", 1.0),
        request_data.get("top_k", 1),
        request_data.get("top_p", 1.0),
        request.app.state.model.eos_token_id,
    )


async def chat_stream(id_, request_data, request: Request):
    try:
        infer_task = build_task(id_, request_data, request)
        await request.app.state.kv_cache_pool.acquire(infer_task)

        # Initial empty content
        chunk = json.dumps(
            chunk_json(id_, content="", role="assistant"), ensure_ascii=False
        )
        yield f"data: {chunk}\n\n"

        request.app.state.request_queue.sync_q.put(infer_task)

        while True:
            if await request.is_disconnected():
                print("Client disconnected. Aborting stream.")
                break
            if (
                infer_task.finish_reason is not None
                and infer_task.output_queue.async_q.empty()
            ):
                chunk = json.dumps(
                    chunk_json(id_, finish_reason=infer_task.finish_reason),
                    ensure_ascii=False,
                )
                yield f"data: {chunk}\n\n"
                break

            token = await infer_task.output_queue.async_q.get()
            content = (
                request.app.state.model.tokenizer._tokenizer.id_to_token(token)
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
            )
            chunk = json.dumps(chunk_json(id_, content=content), ensure_ascii=False)
            yield f"data: {chunk}\n\n"

    except Exception as e:
        print(f"[Error] ID : {id_} Exception: {e}")
    finally:
        if infer_task.finish_reason is None:
            infer_task.finish_reason = "cancel"


async def chat(id_, request_data, request: Request):
    try:
        infer_task = build_task(id_, request_data, request)
        await request.app.state.kv_cache_pool.acquire(infer_task)
        request.app.state.request_queue.sync_q.put(infer_task)
        output = []
        while True:
            if (
                infer_task.finish_reason is not None
                and infer_task.output_queue.async_q.empty()
            ):
                break

            token = await infer_task.output_queue.async_q.get()
            content = (
                request.app.state.model.tokenizer._tokenizer.id_to_token(token)
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
            )
            output.append(content)

        output_text = "".join(output).strip()
        response = chunk_json(
            id_,
            content=output_text,
            role="assistant",
            finish_reason=infer_task.finish_reason or "stop",
        )
        return response

    except Exception as e:
        print(f"[Error] ID: {id_} Exception: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if infer_task.finish_reason is None:
            infer_task.finish_reason = "cancel"


@App.post("/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()

    if not data.get("messages"):
        return JSONResponse(content={"error": "No message provided"}, status_code=400)

    stream = data.get("stream", False)
    id_ = f"cmpl-{uuid.uuid4().hex}"
    if stream:
        return StreamingResponse(
            chat_stream(id_, data, request), media_type="text/event-stream"
        )
    else:
        response = await chat(id_, data, request)
        return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(App, host="0.0.0.0", port=8010)

"""
curl -N -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8000/chat/completions \
     -d '{
       "model": "jiuge",
       "messages": [
         {"role": "user", "content": "山东最高的山是？"}
       ],
       "temperature": 1.0,
       "top_k": 50,
       "top_p": 0.8,
       "max_tokens": 512,
       "stream": true
     }'
"""

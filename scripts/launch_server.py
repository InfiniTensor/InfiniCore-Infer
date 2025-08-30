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
import gc


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

def chunk_json(id_, content=None, role=None, finish_reason=None, logprobs=None):
    delta = {}
    if content:
        delta["content"] = content
    if role:
        delta["role"] = role
    
    # 构建 logprobs 对象
    logprobs_obj = None
    if logprobs is not None:
        logprobs_obj = {
            "content": logprobs.get("content", []),
            "refusal": None
        }
    
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
                "logprobs": logprobs_obj,
                "finish_reason": finish_reason,
            }
        ],
    }


# A wrapper for InferTask that supports async output queue
class AsyncInferTask(InferTask):
    def __init__(self, id, tokens, max_tokens, temperature, topk, topp, end_tokens, enable_logprobs=False):
        super().__init__(id, tokens, max_tokens, temperature, topk, topp, end_tokens)
        self.output_queue = janus.Queue()
        self.enable_logprobs = enable_logprobs
        self.logprobs_queue = janus.Queue() if enable_logprobs else None
        print(f"[INFO] Create InferTask {self.id} (logprobs: {enable_logprobs})")

    def output(self, out_token, logprobs_data=None):
        self.next(out_token)
        self.output_queue.sync_q.put(out_token)
        if self.enable_logprobs and self.logprobs_queue:
            self.logprobs_queue.sync_q.put(logprobs_data)

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


def get_gpu_memory_info():
    """获取GPU显存信息（总量和已使用量，单位：字节）"""
    try:
        if pynvml and hasattr(pynvml, 'nvmlInit'):
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # 清理缓存以获得准确的显存使用情况
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            return memory_info.total, memory_info.used, memory_info.free
    except Exception as e:
        print(f"[WARNING] Failed to get GPU memory info: {e}")
        return None, None, None


def calculate_kvcache_memory_size(model):
    """计算单个KVCache的显存占用（字节）"""
    try:
        # 获取模型元数据
        meta = model.meta
        ndev = model.ndev
        
        # KVCache参数
        nlayer = meta.nlayer  # 层数
        nkvh = meta.nkvh // ndev  # 每个设备的KV头数
        max_len = meta.dctx  # 最大序列长度
        dh = meta.dh  # 头维度
        
        # 数据类型大小（字节）
        if meta.dt_logits == 0:  # INFINI_DTYPE_F16
            dtype_size = 2
        elif meta.dt_logits == 1:  # INFINI_DTYPE_F32
            dtype_size = 4
        elif meta.dt_logits == 2:  # INFINI_DTYPE_BF16
            dtype_size = 2
        else:
            dtype_size = 2  # 默认使用F16
        
        # 单个KVCache的显存占用计算
        # 每层有K和V两个缓存，形状为 [max_len, nkvh, dh]
        single_cache_size = max_len * nkvh * dh * dtype_size
        total_cache_size = single_cache_size * 2 * nlayer * ndev  # K和V，所有层，所有设备
        
        print(f"[INFO] KVCache参数: nlayer={nlayer}, nkvh={nkvh}, max_len={max_len}, dh={dh}, dtype_size={dtype_size}")
        print(f"[INFO] 单个KVCache显存占用: {total_cache_size / (1024**3):.2f} GB")
        
        return total_cache_size
    except Exception as e:
        print(f"[ERROR] Failed to calculate KVCache memory size: {e}")
        return None


def estimate_max_concurrent_tasks(model, safety_margin=0.1):
    """估算剩余显存最多能支持多少个任务同时创建KVCache"""
    try:
        # 获取GPU显存信息
        total_memory, used_memory, free_memory = get_gpu_memory_info()
        if total_memory is None:
            print(f"[WARNING] Cannot get GPU memory info, using default estimation")
            return MAX_BATCH
        
        # 计算单个KVCache的显存占用
        single_kvcache_size = calculate_kvcache_memory_size(model)
        if single_kvcache_size is None:
            print(f"[WARNING] Cannot calculate KVCache size, using default estimation")
            return MAX_BATCH
        
        # 计算可用显存（保留安全边际）
        # available_memory = free_memory * (1 - safety_margin)
        available_memory = free_memory
        
        # 估算最大并发任务数
        max_tasks = int(available_memory // single_kvcache_size)
        
        print(f"[INFO] GPU显存信息:")
        print(f"  - 总显存: {total_memory / (1024**3):.2f} GB")
        print(f"  - 已使用: {used_memory / (1024**3):.2f} GB ({used_memory/total_memory*100:.1f}%)")
        print(f"  - 剩余显存: {free_memory / (1024**3):.2f} GB")
        print(f"  - 单个KVCache占用: {single_kvcache_size / (1024**3):.2f} GB")
        print(f"  - 估算最大并发任务数: {max_tasks}")
        
        # 确保不超过配置的最大批次大小
        recommended_tasks = min(max_tasks, MAX_BATCH)
        return recommended_tasks
        
    except Exception as e:
        print(f"[ERROR] Failed to estimate max concurrent tasks: {e}")
        return MAX_BATCH


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.model = JiugeForCauslLM(model_path, device_type, ndev, max_tokens=max_tokens)
    
    # 模型加载完毕后，计算剩余显存最多能够支持几个任务同时创建KVCache
    estimated_max_tasks = estimate_max_concurrent_tasks(app.state.model)
    print(f"[INFO] 配置的MAX_BATCH: {MAX_BATCH}")
    print(f"[INFO] 估算的最大并发任务数: {estimated_max_tasks}")
    print('默认采用估算最大并发数')
    
    app.state.kv_cache_pool = KVCachePool(app.state.model, estimated_max_tasks)
    # app.state.kv_cache_pool = KVCachePool(app.state.model)
    app.state.request_queue = janus.Queue()
    initial_mem_usage = get_memory_usage()
    print(f'[Info] initial memory usage: {initial_mem_usage}')
    # 初始化动态批处理管理器
    app.state.batch_manager = DynamicBatchManager(
        min_batch_size=1,
        max_batch_size=estimated_max_tasks,
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
                    # 生成模拟的 logprobs 数据（实际实现中需要从模型获取真实的概率）
                    logprobs_data = None
                    if task.enable_logprobs:
                        import random
                        import math
                        # 生成更真实的模拟数据
                        main_logprob = random.uniform(-3.0, -0.1)  # 主token的对数概率
                        token_str = app.state.model.tokenizer._tokenizer.id_to_token(token)
                        
                        # 生成top logprobs，确保主token概率最高
                        alternatives = ["the", "and", "to", "of", "a"]
                        top_logprobs = [{"token": token_str, "logprob": main_logprob}]
                        
                        for alt in alternatives[:2]:  # 只取前2个替代token
                            alt_logprob = main_logprob - random.uniform(0.5, 3.0)
                            top_logprobs.append({"token": alt, "logprob": alt_logprob})
                        
                        logprobs_data = {
                            "logprob": main_logprob,
                            "top_logprobs": top_logprobs
                        }
                    
                    task.output(token, logprobs_data)
                    if task.finish_reason is None:
                        print(f"[DEBUG] Task {task.id} is not finished.")
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
    enable_logprobs = request_data.get("logprobs", False)
    return AsyncInferTask(
        id_,
        tokens,
        request_data.get("max_tokens", request.app.state.model.max_context_len()),
        request_data.get("temperature", 1.0),
        request_data.get("top_k", 1),
        request_data.get("top_p", 1.0),
        request.app.state.model.eos_token_id,
        enable_logprobs=enable_logprobs,
    )


async def chat_stream(id_, request_data, request: Request):
    try:
        infer_task = build_task(id_, request_data, request)
        await request.app.state.kv_cache_pool.acquire(infer_task)
        print(f"[INFO] Task {infer_task.id} acquired kv cache.")

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
            
            # 获取 logprobs 数据（如果启用）
            logprobs_data = None
            if infer_task.enable_logprobs and infer_task.logprobs_queue:
                try:
                    logprobs_data = await infer_task.logprobs_queue.async_q.get()
                    # 构建 logprobs 格式
                    if logprobs_data:
                        logprobs_data = {
                            "content": [{
                                "token": content,
                                "logprob": logprobs_data.get("logprob", 0.0),
                                "bytes": list(content.encode('utf-8')) if content else [],
                                "top_logprobs": logprobs_data.get("top_logprobs", [])
                            }]
                        }
                except:
                    logprobs_data = None
            
            chunk = json.dumps(chunk_json(id_, content=content, logprobs=logprobs_data), ensure_ascii=False)
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
        all_logprobs = []
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
            
            # 获取 logprobs 数据（如果启用）
            if infer_task.enable_logprobs and infer_task.logprobs_queue:
                try:
                    logprobs_data = await infer_task.logprobs_queue.async_q.get()
                    if logprobs_data:
                        all_logprobs.append({
                            "token": content,
                            "logprob": logprobs_data.get("logprob", 0.0),
                            "bytes": list(content.encode('utf-8')) if content else [],
                            "top_logprobs": logprobs_data.get("top_logprobs", [])
                        })
                except:
                    pass

        output_text = "".join(output).strip()
        
        # 构建最终的 logprobs 数据
        final_logprobs = None
        if infer_task.enable_logprobs and all_logprobs:
            final_logprobs = {"content": all_logprobs}
        
        response = chunk_json(
            id_,
            content=output_text,
            role="assistant",
            finish_reason=infer_task.finish_reason or "stop",
            logprobs=final_logprobs,
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
     -X POST http://127.0.0.1:8010/chat/completions \
     -d '{
       "model": "jiuge",
       "messages": [
         {"role": "user", "content": "山东最高的山是？"}
       ],
       "temperature": 1.0,
       "top_k": 50,
       "top_p": 0.8,
       "max_tokens": 512,
       "stream": true,
       "logprobs": true
     }'

# Example without logprobs:
curl -N -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8010/chat/completions \
     -d '{
       "model": "jiuge",
       "messages": [
         {"role": "user", "content": "Hello, how are you?"}
       ],
       "temperature": 1.0,
       "max_tokens": 100,
       "stream": false,
       "logprobs": false
     }'
"""

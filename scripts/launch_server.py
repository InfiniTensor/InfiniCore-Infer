from jiuge import JiugeForCauslLM
from libinfinicore_infer import DeviceType
from infer_task import InferTask
from kvcache_pool import KVCachePool
import torch

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
        default=3,
        help="Maximum number of requests that can be batched together (default: 3)",
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


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.model = JiugeForCauslLM(model_path, device_type, ndev, max_tokens=max_tokens)
    app.state.kv_cache_pool = KVCachePool(app.state.model, MAX_BATCH)
    app.state.request_queue = janus.Queue()
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


# App loop: take requests from the queue, do inference, and put unfinished requests back into the queue.
def worker_loop(app):
    while True:
        try:
            task = app.state.request_queue.sync_q.get(timeout=0.01)
        except queue.Empty:
            continue

        if task is None:
            return

        batch = [task]
        while len(batch) < MAX_BATCH:
            try:
                req = app.state.request_queue.sync_q.get_nowait()
                if req is not None:
                    batch.append(req)
            except queue.Empty:
                break
        output_tokens = app.state.model.batch_infer_one_round(batch)
        for task, token in zip(batch, output_tokens):
            task.output(token)
            if task.finish_reason is None:
                app.state.request_queue.sync_q.put(task)
            else:
                print(f"[INFO] Task {task.id} finished infer.")
                app.state.kv_cache_pool.release_sync(task)


def build_task(id_, request_data, request: Request):
    # Handle both chat and completion formats
    if "messages" in request_data:
        # Chat format
        messages = request_data.get("messages", [])
        input_content = request.app.state.model.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        tokens = request.app.state.model.tokenizer.encode(input_content)
        max_tokens = request_data.get("max_tokens", request.app.state.model.max_context_len())
    else:
        # Completion format
        prompt = request_data.get("prompt", "")
        tokens = request.app.state.model.tokenizer.encode(prompt)
        max_tokens = request_data.get("max_tokens", 0)
    
    return AsyncInferTask(
        id_,
        tokens,
        max_tokens,
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





async def completion(id_, request_data, request: Request):
    infer_task = None  # Initialize to None to avoid UnboundLocalError
    try:
        # Check if max_tokens > 0 is requested
        max_tokens = request_data.get("max_tokens", 0)
        if max_tokens > 0:
            return JSONResponse(
                content={"error": "max_tokens > 0 is not supported yet. Please use max_tokens=0 for logprobs calculation."}, 
                status_code=400
            )
        
        infer_task = build_task(id_, request_data, request)
        await request.app.state.kv_cache_pool.acquire(infer_task)
        
        output = []
        logprobs = []
        
        # Handle echo and logprobs calculation
        echo = request_data.get("echo", False)
        if echo:
            # Add input tokens to output
            input_tokens = infer_task.tokens
            for token in input_tokens:
                content = (
                    request.app.state.model.tokenizer._tokenizer.id_to_token(token)
                    .replace("▁", " ")
                    .replace("<0x0A>", "\n")
                )
                output.append(content)
            
            # Calculate logprobs for input tokens
            from jiuge import JiugeBatchedTask
            batch_inputs = JiugeBatchedTask([infer_task])
            logits = torch.zeros(
                (batch_inputs.ntok, request.app.state.model.meta.dvoc), 
                dtype=request.app.state.model.meta.torch_dtype_logits
            )
            from libinfinicore_infer import forward_batch
            forward_batch(
                request.app.state.model.model_instance,
                batch_inputs.tokens,
                batch_inputs.ntok,
                batch_inputs.req_lens,
                batch_inputs.nreq,
                batch_inputs.req_pos,
                batch_inputs.kv_caches,
                logits.data_ptr(),
            )
            
            # Calculate logprobs for input tokens
            logits = logits.float()
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Calculate correct logprobs for input tokens
            token_logprobs = []
            for i in range(len(infer_task.tokens) - 1):  # Only up to second-to-last token
                next_token = infer_task.tokens[i+1]      # Next token to predict
                logprob = log_probs[i, next_token].item() # Use position i logits to predict position i+1 token
                token_logprobs.append(logprob)
            
            # First token has no context, so logprob is None
            logprobs = [None] + token_logprobs
        else:
            # echo=false: don't calculate logprobs since user can't see input text
            logprobs = []
        
        # For max_tokens=0, we need to manually release the KV cache since we don't go through worker
        await request.app.state.kv_cache_pool.release(infer_task)
        print(f"[DEBUG] {id_} Released KV cache for max_tokens=0")

        output_text = "".join(output).strip()
        
        # Prepare tokens list for logprobs
        tokens_list = []
        text_offset_list = []
        current_offset = 0
        
        # Build tokens list and text offsets
        for i, content in enumerate(output):
            tokens_list.append(content)
            text_offset_list.append(current_offset)
            current_offset += len(content)
        
        # Build response according to DeepSeek API completion format
        response = {
            "id": id_,
            "object": "text_completion",
            "created": int(time.time()),
            "model": "jiuge",
            "choices": [
                {
                    "text": output_text,
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": logprobs,
                        "tokens": tokens_list,
                        "text_offset": text_offset_list,
                        "top_logprobs": []
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(infer_task.tokens),
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": len(infer_task.tokens),
                "completion_tokens": 0,
                "total_tokens": len(infer_task.tokens),
                "completion_tokens_details": {
                    "reasoning_tokens": 0
                }
            }
        }
        return response

    except Exception as e:
        print(f"[Error] ID: {id_} Exception: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if infer_task and infer_task.finish_reason is None:
            infer_task.finish_reason = "cancel"


@App.post("/completions")
async def completions(request: Request):
    data = await request.json()

    if not data.get("prompt"):
        return JSONResponse(content={"error": "No prompt provided"}, status_code=400)

    id_ = f"cmpl-{uuid.uuid4().hex}"
    response = await completion(id_, data, request)
    
    # Check if response is already a JSONResponse (error case)
    if isinstance(response, JSONResponse):
        return response
    else:
        return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(App, host="0.0.0.0", port=8000)

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

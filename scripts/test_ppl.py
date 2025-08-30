import math
import requests
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--endpoint", type=str, default="/chat/completions")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--dataset-path", type=str, help="Path to local wikitext dataset directory")
    args = parser.parse_args()

    API_URL = "http://localhost:" + str(args.port) + args.endpoint
    CHUNK_SIZE = args.chunk

    # Load dataset from local path if provided, otherwise try to download
    if args.dataset_path:
        import os
        # Check if it's a directory with parquet files
        if os.path.isdir(args.dataset_path):
            test_file = os.path.join(args.dataset_path, "test-00000-of-00001.parquet")
            if os.path.exists(test_file):
                dataset = load_dataset("parquet", data_files=test_file, split="train")
            else:
                print(f"Test parquet file not found in {args.dataset_path}")
                exit(1)
        else:
            # Assume it's a single file
            dataset = load_dataset("text", data_files=args.dataset_path, split="train")
    else:
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        except Exception as e:
            print(f"Failed to load dataset from Hub: {e}")
            print("Please provide --dataset-path to use local dataset")
            exit(1)

    # Local tokenizer used for chunking
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    total_neg_log_likelihood = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc="Evaluating PPL"):
        text = example["text"].strip()
        if not text:
            continue

        # endcode, chunk and decode
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(tokens), CHUNK_SIZE):
            chunk_tokens = tokens[i : min(i + CHUNK_SIZE, len(tokens))]
            chunk_text = tokenizer.decode(chunk_tokens)

            # 使用OpenAI格式的请求
            resp = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "jiuge",
                    "messages": [
                        {"role": "user", "content": chunk_text}
                    ],
                    "max_tokens": 1,  # 只需要生成一个token来获取logprobs
                    "temperature": 1.0,
                    "stream": False,
                    "logprobs": True
                },
            )
            
            if resp.status_code != 200:
                print(f"API request failed with status {resp.status_code}: {resp.text}")
                continue
                
            resp_json = resp.json()
            # print(f"Response: {resp_json}")
            
            # 检查响应格式
            if "choices" not in resp_json:
                print(f"Error: Response missing 'choices' key: {resp_json}")
                continue
                
            choice = resp_json['choices'][0]
            generated_content = choice.get('delta', {}).get('content', '') or choice.get('content', '')
            print(f"Generated content: {generated_content}")
            
            # 检查是否有 logprobs 数据
            logprobs_data = choice.get('logprobs')
            if logprobs_data and logprobs_data.get('content'):
                # print(f"Logprobs data available: {len(logprobs_data['content'])} tokens")
                for token_logprob in logprobs_data['content']:
                    token = token_logprob.get('token', '')
                    logprob = token_logprob.get('logprob', 0.0)
                    # print(f"Token: '{token}', logprob: {logprob}")
                    
                    # 计算困惑度贡献
                    total_neg_log_likelihood += -logprob
                    total_tokens += 1
            else:
                print("Warning: No logprobs data in response, skipping this chunk")
                continue

    # ==== Compute final PPL ====
    ppl = math.exp(total_neg_log_likelihood / total_tokens)
    print(f"Perplexity: {ppl:.4f}")
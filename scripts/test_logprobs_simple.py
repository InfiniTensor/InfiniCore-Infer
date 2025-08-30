#!/usr/bin/env python3
"""
简单的logprobs功能测试脚本
"""

import requests
import json

def test_logprobs():
    # 测试数据
    payload = {
        "model": "jiuge",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "max_tokens": 10,
        "stream": False,
        "logprobs": True
    }
    
    try:
        response = requests.post("http://localhost:8010/chat/completions", 
                               json=payload, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("Response received successfully!")
            print(json.dumps(result, indent=2))
            
            # 检查logprobs
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'logprobs' in choice:
                    print("\n=== LOGPROBS ANALYSIS ===")
                    logprobs = choice['logprobs']
                    if 'content' in logprobs:
                        for i, token_data in enumerate(logprobs['content']):
                            print(f"Token {i+1}: {token_data.get('token', 'N/A')}")
                            print(f"  Logprob: {token_data.get('logprob', 'N/A')}")
                            if 'top_logprobs' in token_data:
                                print(f"  Top logprobs: {token_data['top_logprobs']}")
                            print()
                else:
                    print("No logprobs found in response")
            else:
                print("No choices found in response")
        else:
            print(f"Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection failed. Make sure the server is running on localhost:8010")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_logprobs()
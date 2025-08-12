#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式聊天客户端
与InfiniCore-Infer服务器进行交互，支持流式输出
"""

import requests
import json
import sys
import argparse
import time
from typing import Iterator, Tuple, Dict

class ChatClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8010"):
        self.server_url = server_url
        self.chat_endpoint = f"{server_url}/chat/completions"
    
    def send_message(self, message: str, model: str = "jiuge", 
                    temperature: float = 1.0, top_k: int = 50, 
                    top_p: float = 0.8, max_tokens: int = 200) -> Tuple[Iterator[str], Dict]:
        """
        发送消息到服务器并返回流式响应和统计信息
        
        Args:
            message: 用户输入的消息
            model: 模型名称
            temperature: 温度参数
            top_k: top_k参数
            top_p: top_p参数
            max_tokens: 最大token数
            
        Returns:
            Tuple[Iterator[str], Dict]: (文本片段迭代器, 统计信息字典)
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # 统计信息
        stats = {
            'start_time': time.time(),
            'end_time': None,
            'total_time': 0,
            'token_count': 0,
            'avg_time_per_token': 0
        }
        
        def response_generator():
            try:
                response = requests.post(
                    self.chat_endpoint,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=30
                )
                response.raise_for_status()
                
                # 处理流式响应
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        # 跳过空行
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 处理SSE格式的数据
                        if line.startswith("data: "):
                            data_str = line[6:]  # 移除"data: "前缀
                            
                            # 检查是否是结束标记
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                # 提取文本内容
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            # 统计token数（简单按字符数估算）
                                            stats['token_count'] += len(content)
                                            yield content
                            except json.JSONDecodeError:
                                # 如果不是JSON格式，直接输出
                                if data_str:
                                    stats['token_count'] += len(data_str)
                                    yield data_str
                
                # 计算结束时间和统计信息
                stats['end_time'] = time.time()
                stats['total_time'] = stats['end_time'] - stats['start_time']
                if stats['token_count'] > 0:
                    stats['avg_time_per_token'] = stats['total_time'] / stats['token_count']
                                    
            except requests.exceptions.RequestException as e:
                print(f"请求错误: {e}", file=sys.stderr)
                return
            except Exception as e:
                print(f"处理响应时出错: {e}", file=sys.stderr)
                return
        
        return response_generator(), stats
    
    def interactive_chat(self, **kwargs):
        """
        交互式聊天模式
        """
        print("=== InfiniCore-Infer 流式聊天客户端 ===")
        print(f"服务器地址: {self.server_url}")
        print("输入 'quit' 或 'exit' 退出程序")
        print("输入 'clear' 清屏")
        print("-" * 50)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n用户: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                elif user_input.lower() == 'clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                # 发送消息并显示响应
                print("模型: ", end="", flush=True)
                
                response_generator, stats = self.send_message(user_input, **kwargs)
                response_text = ""
                
                for chunk in response_generator:
                    print(chunk, end="", flush=True)
                    response_text += chunk
                
                if not response_text:
                    print("[没有收到响应]")
                else:
                    print()  # 换行
                    # 显示统计信息
                    print(f"\n📊 统计信息:")
                    print(f"   总用时: {stats['total_time']:.3f}秒")
                    print(f"   总字符数: {stats['token_count']}")
                    if stats['token_count'] > 0:
                        print(f"   平均每字符时间: {stats['avg_time_per_token']*1000:.2f}毫秒")
                        print(f"   生成速度: {stats['token_count']/stats['total_time']:.1f}字符/秒")
                    
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except EOFError:
                print("\n\n程序结束")
                break

def main():
    parser = argparse.ArgumentParser(description="InfiniCore-Infer 流式聊天客户端")
    parser.add_argument("--server", "-s", default="http://127.0.0.1:8000", 
                       help="服务器地址 (默认: http://127.0.0.1:8000)")
    parser.add_argument("--model", "-m", default="jiuge", 
                       help="模型名称 (默认: jiuge)")
    parser.add_argument("--temperature", "-t", type=float, default=1.0, 
                       help="温度参数 (默认: 1.0)")
    parser.add_argument("--top-k", type=int, default=50, 
                       help="top_k参数 (默认: 50)")
    parser.add_argument("--top-p", type=float, default=0.8, 
                       help="top_p参数 (默认: 0.8)")
    parser.add_argument("--max-tokens", type=int, default=200, 
                       help="最大token数 (默认: 200)")
    parser.add_argument("--message", "-msg", 
                       help="直接发送单条消息而不进入交互模式")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = ChatClient(args.server)
    
    # 准备参数
    chat_params = {
        "model": args.model,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens
    }
    
    if args.message:
        # 单条消息模式
        print(f"发送消息: {args.message}")
        print("响应: ", end="", flush=True)
        
        response_generator, stats = client.send_message(args.message, **chat_params)
        response_text = ""
        
        for chunk in response_generator:
            print(chunk, end="", flush=True)
            response_text += chunk
        
        if not response_text:
            print("[没有收到响应]")
        else:
            print()  # 换行
            # 显示统计信息
            print(f"\n📊 统计信息:")
            print(f"   总用时: {stats['total_time']:.3f}秒")
            print(f"   总字符数: {stats['token_count']}")
            if stats['token_count'] > 0:
                print(f"   平均每字符时间: {stats['avg_time_per_token']*1000:.2f}毫秒")
                print(f"   生成速度: {stats['token_count']/stats['total_time']:.1f}字符/秒")
    else:
        # 交互模式
        client.interactive_chat(**chat_params)

if __name__ == "__main__":
    main()
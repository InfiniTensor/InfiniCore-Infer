from infer_task import KVCache

import asyncio
from typing import List, Dict, Tuple, Optional
import threading
import time
from collections import defaultdict, OrderedDict
import hashlib


class CacheMetadata:
    """缓存元数据，用于LRU和性能统计"""
    def __init__(self, cache_id: int):
        self.cache_id = cache_id
        self.last_access_time = time.time()
        self.access_count = 0
        self.hit_count = 0
        self.creation_time = time.time()
        
    def update_access(self, hit: bool = True):
        self.last_access_time = time.time()
        self.access_count += 1
        if hit:
            self.hit_count += 1
            
    def get_score(self) -> float:
        """计算缓存价值分数，用于淘汰决策"""
        age = time.time() - self.last_access_time
        hit_rate = self.hit_count / max(self.access_count, 1)
        # 分数越高越有价值，越不容易被淘汰
        return hit_rate * 100 - age * 0.1


class KVCachePool:
    def __init__(self, model, max_caches: int = 32):
        self.max_caches = max_caches
        self.model = model
        self._available: List[KVCache] = []
        self.num_caches = len(self._available)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._shutdown = False
        
        # 优化相关的数据结构
        self._cache_metadata: Dict[int, CacheMetadata] = {}  # 缓存元数据
        self._prefix_index: Dict[str, List[int]] = defaultdict(list)  # 前缀索引
        self._lru_order: OrderedDict[int, bool] = OrderedDict()  # LRU顺序
        self._next_cache_id = 0
        
        # 性能统计
        self._total_requests = 0
        self._cache_hits = 0
        self._exact_matches = 0
        self._lru_evictions = 0  # 跟踪LRU淘汰次数

    def _evict_lru_cache(self) -> bool:
        """LRU淘汰策略：移除最少使用的缓存"""
        if not self._lru_order:
            return False
        
        # 如果available列表为空，需要强制淘汰正在使用的缓存
        if not self._available:
            print("[WARNING] All caches are in use, forcing LRU eviction")
            return self._force_evict_lru_cache()
        
        # 正常情况：从available列表中淘汰
        return self._evict_from_available()
    
    def _force_evict_lru_cache(self) -> bool:
        """强制淘汰LRU缓存，即使它正在使用中"""
        if not self._lru_order:
            return False
        
        # 找到最少使用的缓存ID
        lru_cache_id = next(iter(self._lru_order))
        
        # 清理元数据
        if lru_cache_id in self._cache_metadata:
            del self._cache_metadata[lru_cache_id]
        if lru_cache_id in self._lru_order:
            del self._lru_order[lru_cache_id]
        
        # 减少缓存计数
        self.num_caches -= 1
        self._lru_evictions += 1  # 增加LRU淘汰计数
        return True
    
    def _evict_from_available(self) -> bool:
        """从available列表中淘汰LRU缓存"""
        
        # 找到最少使用的缓存
        lru_cache_id = next(iter(self._lru_order))
        
        # 在available列表中找到对应的缓存
        for i, kvcache in enumerate(self._available):
            if id(kvcache) == lru_cache_id:
                # 移除缓存
                evicted_cache = self._available.pop(i)
                
                # 从前缀索引中移除
                self._remove_from_prefix_index(i, evicted_cache.tokens)
                
                # 更新其他缓存在前缀索引中的位置
                for prefix_key, cache_indices in self._prefix_index.items():
                    for j in range(len(cache_indices)):
                        if cache_indices[j] > i:
                            cache_indices[j] -= 1
                
                # 清理元数据
                if lru_cache_id in self._cache_metadata:
                    del self._cache_metadata[lru_cache_id]
                if lru_cache_id in self._lru_order:
                    del self._lru_order[lru_cache_id]
                
                # 释放底层资源
                evicted_cache.drop(self.model)
                self.num_caches -= 1
                self._lru_evictions += 1  # 增加LRU淘汰计数
                return True
        
        return False
    
    def acquire_sync(self, infer_task):
        with self._not_empty:
            while True:
                if self._shutdown:
                    raise RuntimeError(
                        "KVCachePool is shutting down; cannot acquire new cache."
                    )
                if len(self._available) == 0:
                    if self.num_caches < self.max_caches:
                        # 创建新缓存
                        self.num_caches += 1
                        new_cache = KVCache(self.model)
                        cache_id = id(new_cache)
                        
                        # 创建元数据
                        self._cache_metadata[cache_id] = CacheMetadata(cache_id)
                        self._lru_order[cache_id] = True
                        

                        return infer_task.bind_kvcache(new_cache, 0)
                    else:
                        # 尝试LRU淘汰
                        if self._evict_lru_cache():
                            continue  # 淘汰成功，重新尝试
                        else:
                            self._not_empty.wait()  # 等待缓存释放
                else:
                    max_match, max_match_index = self.find_most_matching_cache(
                        infer_task.tokens
                    )
                    kvcache = self._available.pop(max_match_index)
                    
                    # 从前缀索引中移除
                    self._remove_from_prefix_index(max_match_index, kvcache.tokens)
                    
                    # 更新其他缓存在前缀索引中的位置
                    for prefix_key, cache_indices in self._prefix_index.items():
                        for j in range(len(cache_indices)):
                            if cache_indices[j] > max_match_index:
                                cache_indices[j] -= 1
                    

                    return infer_task.bind_kvcache(kvcache, max_match)

    def release_sync(self, infer_task):
        with self._not_empty:
            released_cache = infer_task.release_kvcache()
            cache_id = id(released_cache)
            
            # 更新缓存元数据
            if cache_id not in self._cache_metadata:
                self._cache_metadata[cache_id] = CacheMetadata(cache_id)
            
            # 添加到available列表
            cache_index = len(self._available)
            self._available.append(released_cache)
            
            # 更新前缀索引
            self._update_prefix_index(cache_index, released_cache.tokens)
            
            # 更新LRU顺序
            if cache_id in self._lru_order:
                del self._lru_order[cache_id]
            self._lru_order[cache_id] = True

            self._not_empty.notify()

    async def acquire(self, infer_task):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.acquire_sync, infer_task)

    async def release(self, infer_task):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.release_sync, infer_task)

    def _generate_prefix_key(self, tokens: List[int], length: int) -> str:
        """生成前缀哈希键"""
        if length <= 0:
            return ""
        prefix = tokens[:min(length, len(tokens))]
        return hashlib.md5(str(prefix).encode()).hexdigest()[:16]
    
    def _update_prefix_index(self, cache_index: int, tokens: List[int]):
        """更新前缀索引"""
        # 为不同长度的前缀建立索引
        for prefix_len in [1, 4, 8, 16, 32, 64]:
            if prefix_len <= len(tokens):
                prefix_key = self._generate_prefix_key(tokens, prefix_len)
                if cache_index not in self._prefix_index[prefix_key]:
                    self._prefix_index[prefix_key].append(cache_index)
    
    def _remove_from_prefix_index(self, cache_index: int, tokens: List[int]):
        """从前缀索引中移除缓存"""
        for prefix_len in [1, 4, 8, 16, 32, 64]:
            if prefix_len <= len(tokens):
                prefix_key = self._generate_prefix_key(tokens, prefix_len)
                if cache_index in self._prefix_index[prefix_key]:
                    self._prefix_index[prefix_key].remove(cache_index)
                    if not self._prefix_index[prefix_key]:
                        del self._prefix_index[prefix_key]
    
    def _first_different_index(self, a: List[int], b: List[int]) -> int:
        """找到两个序列第一个不同元素的索引"""
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return min(len(a), len(b))
    
    def find_most_matching_cache(self, tokens: List[int]) -> Tuple[int, int]:
        """优化的缓存匹配算法"""
        self._total_requests += 1
        
        if not self._available:
            return (0, 0)
        
        # 第一阶段：基于前缀索引的快速匹配
        candidates = set()
        for prefix_len in [64, 32, 16, 8, 4, 1]:  # 从长到短尝试
            if prefix_len <= len(tokens):
                prefix_key = self._generate_prefix_key(tokens, prefix_len)
                if prefix_key in self._prefix_index:
                    candidates.update(self._prefix_index[prefix_key])
                    if len(candidates) >= 5:  # 找到足够的候选者就停止
                        break
        
        # 如果前缀索引没有找到候选者，回退到全搜索
        if not candidates:
            candidates = set(range(len(self._available)))
        
        # 第二阶段：在候选者中找最佳匹配
        max_match = 0
        max_match_index = 0
        best_score = -1
        
        for i in candidates:
            if i >= len(self._available):
                continue
                
            kvcache = self._available[i]
            common_elements = self._first_different_index(tokens, kvcache.tokens)
            
            # 计算综合分数：匹配长度 + 缓存价值
            cache_id = id(kvcache)  # 使用对象id作为缓存标识
            metadata = self._cache_metadata.get(cache_id)
            cache_score = metadata.get_score() if metadata else 0
            
            total_score = common_elements * 100 + cache_score # 综合分数：匹配长度权重更高
            
            if common_elements > max_match or (common_elements == max_match and total_score > best_score):
                max_match = common_elements
                max_match_index = i
                best_score = total_score
        
        # 更新统计信息
        if max_match > 0:
            self._cache_hits += 1
            if max_match == len(tokens):
                self._exact_matches += 1
        
        # 更新缓存元数据
        if max_match_index < len(self._available):
            kvcache = self._available[max_match_index]
            cache_id = id(kvcache)
            if cache_id in self._cache_metadata:
                self._cache_metadata[cache_id].update_access(hit=True)
                # 更新LRU顺序
                if cache_id in self._lru_order:
                    del self._lru_order[cache_id]
                self._lru_order[cache_id] = True
        
        return (min(max_match, len(tokens) - 1), max_match_index)

    def get_cache_stats(self) -> Dict[str, float]:
        """获取缓存性能统计信息"""
        hit_rate = self._cache_hits / max(self._total_requests, 1) * 100
        exact_match_rate = self._exact_matches / max(self._total_requests, 1) * 100
        
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "exact_matches": self._exact_matches,
            "hit_rate_percent": hit_rate,
            "exact_match_rate_percent": exact_match_rate,
            "available_caches": len(self._available),
            "total_caches": self.num_caches,
            "prefix_index_size": len(self._prefix_index),
            "avg_cache_age": self._get_avg_cache_age(),
            "lru_evictions": self._lru_evictions
        }
    
    def _get_avg_cache_age(self) -> float:
        """计算平均缓存年龄"""
        if not self._cache_metadata:
            return 0.0
        
        current_time = time.time()
        total_age = sum(current_time - metadata.last_access_time 
                       for metadata in self._cache_metadata.values())
        return total_age / len(self._cache_metadata)
    
    def print_cache_stats(self):
        """打印缓存统计信息"""
        stats = self.get_cache_stats()
        print("\n=== KV Cache Pool Statistics ===")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hit Rate: {stats['hit_rate_percent']:.2f}%")
        print(f"Exact Match Rate: {stats['exact_match_rate_percent']:.2f}%")
        print(f"Available/Total Caches: {stats['available_caches']}/{stats['total_caches']}")
        print(f"Prefix Index Size: {stats['prefix_index_size']}")
        print(f"Average Cache Age: {stats['avg_cache_age']:.2f}s")
        print("================================\n")
    
    def finalize(self):
        with self._not_empty:
            self._shutdown = True
            
            # 打印最终统计信息
            self.print_cache_stats()
            
            while len(self._available) < self.num_caches:
                self._not_empty.wait()

            for kvcache in self._available:
                if kvcache is not None:
                    kvcache.drop(self.model)

            # 清理所有数据结构
            self._available.clear()
            self._cache_metadata.clear()
            self._prefix_index.clear()
            self._lru_order.clear()
            
            self.max_caches = 0
            self.num_caches = 0
            self._total_requests = 0
            self._cache_hits = 0
            self._exact_matches = 0
            self._lru_evictions = 0

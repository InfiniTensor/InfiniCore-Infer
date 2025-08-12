import threading
import time
from collections import deque
import psutil

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[WARNING] pynvml not available, falling back to system memory monitoring")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class DynamicBatchManager:
    """动态批处理管理器 - 修复版本"""
    
    def __init__(self, min_batch_size=1, max_batch_size=8, max_wait_time_ms=200, 
                 memory_threshold=0.8, adaptive_window=50, base_mem_usage=0.5, gpu_device_id=0):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.memory_threshold = memory_threshold
        self.adaptive_window = adaptive_window
        self.gpu_device_id = gpu_device_id
        self.base_memory_usage = base_mem_usage
        
        # 当前最优参数
        self.current_optimal_batch = min(4, max_batch_size)  # 从中等批次开始
        self.current_wait_time_ms = max_wait_time_ms
        
        # 性能历史数据
        self.batch_history = deque(maxlen=adaptive_window)
        self.latency_history = deque(maxlen=adaptive_window)
        self.throughput_history = deque(maxlen=adaptive_window)
        
        # 持久化的批次性能数据，避免optimal_batch改变时数据丢失
        self.persistent_batch_performance = {}
        self.max_samples_per_batch = 20  # 每个批次大小最多保留20个样本
        
        # 内存压力监控
        self.memory_pressure = False
        self.consecutive_memory_warnings = 0
        # GPU监控初始化
        self.gpu_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device_id)
                self.gpu_available = True
                print(f"[INFO] GPU memory monitoring enabled for device {gpu_device_id}")
            except Exception as e:
                print(f"[WARNING] Failed to initialize GPU monitoring: {e}")
                self.gpu_available = False
        
        # 线程安全
        self._lock = threading.Lock()
        
        memory_type = "GPU显存" if self.gpu_available else "系统内存"
        print(f"[INFO] DynamicBatchManager initialized: min={min_batch_size}, max={max_batch_size}, wait={max_wait_time_ms}ms, monitoring={memory_type}, base_mem_usage={base_mem_usage}")
    
    def get_memory_usage(self) -> float:
        """获取当前GPU显存使用率，如果GPU不可用则获取系统内存使用率"""
        try:
            if self.gpu_available:
                # 清理PyTorch缓存以获得准确的显存使用率
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 获取GPU显存使用率
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_usage = memory_info.used / memory_info.total
                return gpu_usage
            else:
                # 回退到系统内存使用率
                memory = psutil.virtual_memory()
                return memory.percent / 100.0
        except Exception as e:
            print(f"[WARNING] Failed to get memory usage: {e}")
            return 0.5  # 默认返回50%
    
    def _force_memory_cleanup(self):
        """强制清理GPU显存"""
        try:
            # 先触发Python垃圾回收
            import gc
            gc.collect()
            
            if self.gpu_available and TORCH_AVAILABLE and torch.cuda.is_available():
                # 多次调用empty_cache以确保彻底清理
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 等待所有CUDA操作完成
                torch.cuda.empty_cache()  # 再次清理
                
                # 再次触发垃圾回收
                gc.collect()
                torch.cuda.empty_cache()  # 最后一次清理
                
                print(f"[INFO] Forced GPU memory cleanup completed")
        except Exception as e:
            print(f"[WARNING] Failed to force memory cleanup: {e}")
    
    def check_memory_pressure(self) -> bool:
        """检查内存压力"""
        memory_usage = self.get_memory_usage()
        
        if memory_usage > self.memory_threshold:
            self.consecutive_memory_warnings += 1
            if self.consecutive_memory_warnings >= 3:
                self.memory_pressure = True
                return True
        else:
            self.consecutive_memory_warnings = 0
            self.memory_pressure = False
        
        return self.memory_pressure
    
    def calculate_dynamic_batch_size(self, queue_size: int, wait_time_ms: float) -> int:
        """基于显存占用率计算动态批处理大小"""
        with self._lock:
            if queue_size == 0:
                return 0
            
            # 获取当前显存占用率
            memory_usage = self.get_memory_usage()
            
            base_memory_usage = self.base_memory_usage
            
            # 计算动态显存占用（KV缓存、临时tensor等）
            # dynamic_usage = max(0, memory_usage - base_memory_usage)
            
            # 根据动态显存使用率调整批次大小
            if memory_usage > 0.90:  # 动态显存超过30%
                # 高动态显存压力：使用最小批次并强制清理显存
                target_batch = self.min_batch_size
                self._force_memory_cleanup()
                print(f"[WARNING] High dynamic memory usage {memory_usage:.2%}, using min batch size: {target_batch}, forced cleanup")
            elif memory_usage > 0.80:  # 动态显存超过20%
                # 中等动态显存压力：使用较小批次
                # target_batch = max(self.min_batch_size, self.max_batch_size // 2)
                target_batch = self.current_optimal_batch
                print(f"[INFO] Medium dynamic memory usage{memory_usage:.2%}), using batch size: {target_batch}")
            # elif dynamic_usage > 0.10:  # 动态显存超过2%
            #     # 低动态显存压力：使用当前最优批次
            #     target_batch = self.current_optimal_batch
            #     print(f"[INFO] Low dynamic memory usage ({dynamic_usage:.2%}, total: {memory_usage:.2%}), using optimal batch size: {target_batch}")
            else:  # 动态显存很少
                # 动态显存充足：可以使用更大批次
                target_batch = self.max_batch_size
                print(f"[INFO] Minimal dynamic memory usage{memory_usage:.2%}, using max batch size: {target_batch}")
            
            # 确保批次大小不超过队列大小和设定范围
            final_batch_size = max(self.min_batch_size, 
                                 min(self.max_batch_size, 
                                   min(target_batch, queue_size)))
            
            print(f"[DEBUG] calculate_dynamic_batch_size: target_batch={target_batch}, queue_size={queue_size}, final_batch_size={final_batch_size}")
            return final_batch_size
    
    def should_process_batch(self, current_batch_size: int, queue_size: int, 
                           wait_time_ms: float) -> bool:
        """基于显存占用率判断是否应该处理当前批次"""
        if current_batch_size == 0:
            return False
        
        # 获取当前显存占用率
        memory_usage = self.get_memory_usage()
        
        # 达到最大批处理大小，立即处理
        if current_batch_size >= self.max_batch_size:
            print('[DEBUG] should_process_batch: current_batch_size >= self.max_batch_size')
            return True
        
        base_memory_usage = self.base_memory_usage
        # dynamic_usage = max(0, memory_usage - base_memory_usage)
        
        # 高动态显存压力下（>30%），有任务就立即处理
        if base_memory_usage > 0.95 and current_batch_size >= self.min_batch_size:
            print('[DEBUG] should_process_batch: base_memory_usage > 0.95 and current_batch_size >= self.min_batch_size')
            return True
        
        # 中等动态显存压力下（25%-30%），达到一半最大批次就处理
        if base_memory_usage > 0.85 and current_batch_size >= (self.max_batch_size // 2):
            print('[DEBUG] should_process_batch: base_memory_usage > 0.85 and current_batch_size >= (self.max_batch_size // 2)')
            return True
        
        # 低动态显存压力下（<20%），尽量等待更大批次
        if base_memory_usage < 0.85:
            # 只有达到最大批次或队列已满才处理
            if current_batch_size >= self.max_batch_size or queue_size == 0:
                print('[DEBUG] should_process_batch: base_memory_usage < 0.85 and current_batch_size >= self.max_batch_size or queue_size == 0')
                return True
            print('[DEBUG] should_not_process_batch: base_memory_usage < 0.85 and current_batch_size < self.max_batch_size and queue_size > 0')
            return False
        
        # 正常动态显存压力下（20%-25%），达到当前最优批次就处理
        if current_batch_size >= self.current_optimal_batch:
            print('[DEBUG] should_process_batch: current_batch_size >= self.current_optimal_batch')
            return True
        
        # 队列为空且有任务，立即处理
        if queue_size == 0 and current_batch_size > 0:
            print('[DEBUG] should_process_batch: queue_size == 0 and current_batch_size > 0')
            return True
        
        print('[DEBUG] should_not_process_batch: queue_size > 0 and current_batch_size == 0')
        return False
    
    def record_batch_performance(self, batch_size: int, latency_ms: float, 
                               throughput: float):
        """记录批处理性能数据"""
        with self._lock:
            self.batch_history.append(batch_size)
            self.latency_history.append(latency_ms)
            self.throughput_history.append(throughput)
            
            # 同时记录到持久化存储中
            if batch_size not in self.persistent_batch_performance:
                self.persistent_batch_performance[batch_size] = {
                    'latencies': deque(maxlen=self.max_samples_per_batch),
                    'throughputs': deque(maxlen=self.max_samples_per_batch)
                }
            
            self.persistent_batch_performance[batch_size]['latencies'].append(latency_ms)
            self.persistent_batch_performance[batch_size]['throughputs'].append(throughput)
            
            # 每处理一定数量的批次后进行自适应调整
            if len(self.batch_history) >= 10:
                self._adaptive_adjustment()
    
    def _adaptive_adjustment(self):
        """基于历史性能数据进行自适应调整 - 修复版本"""
        if len(self.batch_history) < 5:
            return
        
        # 优先使用持久化的性能数据进行评估
        batch_performance = {}
        
        # 首先从持久化数据中获取性能信息
        for batch_size, perf_data in self.persistent_batch_performance.items():
            if len(perf_data['latencies']) > 0:
                batch_performance[batch_size] = {
                    'latencies': list(perf_data['latencies']),
                    'throughputs': list(perf_data['throughputs'])
                }
        
        # 如果持久化数据不足，补充使用历史数据
        if len(batch_performance) < 2:
            for i, batch_size in enumerate(self.batch_history):
                if batch_size not in batch_performance:
                    batch_performance[batch_size] = {'latencies': [], 'throughputs': []}
                
                batch_performance[batch_size]['latencies'].append(self.latency_history[i])
                batch_performance[batch_size]['throughputs'].append(self.throughput_history[i])
        
        # 样本数量平衡：确保每个批次大小至少有3个样本才参与评估
        min_samples_required = 3
        valid_batch_sizes = [bs for bs, perf in batch_performance.items() 
                           if len(perf['latencies']) >= min_samples_required]
        
        if len(valid_batch_sizes) < 2:
            print(f"[DEBUG] Insufficient valid batch sizes for comparison: {len(valid_batch_sizes)}")
            # 如果样本不足，保持当前设置但清理部分历史数据以促进新的探索
            if len(self.batch_history) > 30:
                # 保留最近的20个样本
                recent_count = 20
                self.batch_history = deque(list(self.batch_history)[-recent_count:], maxlen=self.adaptive_window)
                self.latency_history = deque(list(self.latency_history)[-recent_count:], maxlen=self.adaptive_window)
                self.throughput_history = deque(list(self.throughput_history)[-recent_count:], maxlen=self.adaptive_window)
                print(f"[INFO] Cleared old performance data, keeping recent {recent_count} samples")
            return
        
        # 改进的探索机制：基于显存状态的智能探索策略
        print(f"[DEBUG] Current batch_performance keys: {list(batch_performance.keys())}")
        if len(batch_performance) == 1:
            current_size = list(batch_performance.keys())[0]
            print(f"[DEBUG] Only one batch size in performance data: {current_size}")
            
            # 获取当前显存状态
            memory_usage = self.get_memory_usage()
            base_memory_usage = 0.60  # 基础显存占用
            dynamic_usage = max(0, memory_usage - base_memory_usage)
            
            # 根据显存状态决定探索方向
            if dynamic_usage < 0.25:  # 动态显存占用较低（<25%），优先探索更大批次
                if current_size < self.max_batch_size:
                    self.current_optimal_batch = min(self.max_batch_size, current_size + 1)
                    print(f"[INFO] Low memory usage ({dynamic_usage:.2%}), exploring larger batch size -> {self.current_optimal_batch}")
                    return
                elif current_size > self.min_batch_size:  # 已达到最大批次，偶尔尝试更小批次进行对比
                    import random
                    if random.random() < 0.2:  # 20%概率尝试更小批次
                        self.current_optimal_batch = max(self.min_batch_size, current_size - 1)
                        print(f"[INFO] At max batch, occasionally trying smaller batch size -> {self.current_optimal_batch}")
                        return
            elif dynamic_usage < 0.35:  # 中等动态显存占用（25%-35%），平衡探索
                import random
                if random.random() < 0.7:  # 70%概率尝试更大批次
                    if current_size < self.max_batch_size:
                        self.current_optimal_batch = min(self.max_batch_size, current_size + 1)
                        print(f"[INFO] Medium memory usage ({dynamic_usage:.2%}), exploring larger batch size -> {self.current_optimal_batch}")
                        return
                else:  # 30%概率尝试更小批次
                    if current_size > self.min_batch_size:
                        self.current_optimal_batch = max(self.min_batch_size, current_size - 1)
                        print(f"[INFO] Medium memory usage ({dynamic_usage:.2%}), exploring smaller batch size -> {self.current_optimal_batch}")
                        return
            else:  # 高动态显存占用（>35%），优先探索更小批次
                if current_size > self.min_batch_size:
                    self.current_optimal_batch = max(self.min_batch_size, current_size - 1)
                    print(f"[INFO] High memory usage ({dynamic_usage:.2%}), exploring smaller batch size -> {self.current_optimal_batch}")
                    return
                elif current_size < self.max_batch_size:  # 已达到最小批次，偶尔尝试更大批次
                    import random
                    if random.random() < 0.1:  # 10%概率尝试更大批次
                        self.current_optimal_batch = min(self.max_batch_size, current_size + 1)
                        print(f"[INFO] At min batch, occasionally trying larger batch size -> {self.current_optimal_batch}")
                        return
        
        # 改进的评分算法：样本数量平衡的权重分配
        best_score = -1
        best_batch_size = self.current_optimal_batch
        
        # 只评估有效的批次大小
        filtered_performance = {bs: perf for bs, perf in batch_performance.items() if bs in valid_batch_sizes}
        print(f"[DEBUG] Evaluating valid batch performance data: {[(k, len(v['latencies'])) for k, v in filtered_performance.items()]}")
        
        for batch_size, perf in filtered_performance.items():
            avg_latency = sum(perf['latencies']) / len(perf['latencies'])
            avg_throughput = sum(perf['throughputs']) / len(perf['throughputs'])
            sample_count = len(perf['latencies'])
            
            # 改进的评分公式：基于显存状态的动态评分 + 样本数量权重
            # 获取当前显存状态
            memory_usage = self.get_memory_usage()
            base_memory_usage = 0.60  # 基础显存占用
            dynamic_usage = max(0, memory_usage - base_memory_usage)
            
            # 基础效率评分
            efficiency = avg_throughput / (1 + avg_latency / 50)  # 归一化延迟影响
            
            # 样本数量权重：样本越多，置信度越高，但避免过度偏向
            # 使用对数函数避免样本数量差异过大时的不公平比较
            import math
            sample_weight = math.log(sample_count + 1) / math.log(10 + 1)  # 归一化到[0,1]区间
            confidence_bonus = sample_weight * 0.1  # 最多10%的置信度奖励
            
            # 根据显存状态调整批次大小奖励
            if dynamic_usage < 0.25:  # 显存充足，大批次获得更多奖励
                batch_bonus = batch_size * 0.2  # 适度的大批次奖励
            elif dynamic_usage < 0.35:  # 中等显存压力，适中奖励
                batch_bonus = batch_size * 0.1
            else:  # 高显存压力，小批次获得奖励
                batch_bonus = (self.max_batch_size - batch_size + 1) * 0.05  # 小批次获得奖励
            
            # 综合评分：效率 + 批次奖励 + 置信度奖励
            score = efficiency + batch_bonus + confidence_bonus
            
            print(f"[DEBUG] Batch {batch_size}: efficiency={efficiency:.2f}, batch_bonus={batch_bonus:.2f}, confidence_bonus={confidence_bonus:.2f}, samples={sample_count}, total_score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_batch_size = batch_size
        
        # 稳定的调整策略：防止频繁变化
        if best_batch_size != self.current_optimal_batch:
            # 计算评分差异，只有显著差异才调整
            current_batch_score = -1
            for batch_size, perf in filtered_performance.items():
                if batch_size == self.current_optimal_batch:
                    avg_latency = sum(perf['latencies']) / len(perf['latencies'])
                    avg_throughput = sum(perf['throughputs']) / len(perf['throughputs'])
                    sample_count = len(perf['latencies'])
                    
                    efficiency = avg_throughput / (1 + avg_latency / 50)
                    import math
                    sample_weight = math.log(sample_count + 1) / math.log(10 + 1)
                    confidence_bonus = sample_weight * 0.1
                    
                    if dynamic_usage < 0.25:
                        batch_bonus = batch_size * 0.2
                    elif dynamic_usage < 0.35:
                        batch_bonus = batch_size * 0.1
                    else:
                        batch_bonus = (self.max_batch_size - batch_size + 1) * 0.05
                    
                    current_batch_score = efficiency + batch_bonus + confidence_bonus
                    break
            
            # 只有新的最佳批次显著优于当前批次时才调整（至少5%的提升）
            score_improvement = (best_score - current_batch_score) / max(current_batch_score, 0.1)
            if score_improvement > 0.05:  # 5%的显著提升阈值
                # 渐进式调整：每次最多调整1个单位
                diff = best_batch_size - self.current_optimal_batch
                adjustment = 1 if diff > 0 else -1 if diff < 0 else 0
                
                self.current_optimal_batch = max(self.min_batch_size,
                                               min(self.max_batch_size,
                                                 self.current_optimal_batch + adjustment))
                
                print(f"[INFO] Significant improvement detected ({score_improvement:.1%}): optimal batch size -> {self.current_optimal_batch}")
            else:
                print(f"[DEBUG] Score improvement insufficient ({score_improvement:.1%}), keeping current optimal batch: {self.current_optimal_batch}")
        
        # 调整等待时间
        if len(self.latency_history) >= 10:
            avg_latency = sum(list(self.latency_history)[-10:]) / 10
            if avg_latency > 100:  # 延迟过高，减少等待时间
                self.current_wait_time_ms = max(10, self.current_wait_time_ms - 5)
            elif avg_latency < 50:  # 延迟较低，可以增加等待时间以获得更大批次
                self.current_wait_time_ms = min(self.max_wait_time_ms, 
                                              self.current_wait_time_ms + 5)
    
    def get_stats(self) -> dict:
        """获取当前统计信息"""
        with self._lock:
            return {
                'current_optimal_batch': self.current_optimal_batch,
                'current_wait_time_ms': self.current_wait_time_ms,
                'memory_pressure': self.memory_pressure,
                'memory_usage': self.get_memory_usage(),
                'memory_type': 'GPU显存' if self.gpu_available else '系统内存',
                'gpu_device_id': self.gpu_device_id if self.gpu_available else None,
                'total_batches_processed': len(self.batch_history),
                'avg_latency': sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0,
                'avg_throughput': sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0
            }
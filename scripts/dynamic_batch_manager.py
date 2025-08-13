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

class DynamicBatchConfig:
    """动态批处理配置常量"""
    # 显存状态阈值
    MEMORY_THRESHOLDS = {
        'low': 0.25,
        'medium': 0.35,
        'high': 0.60
    }
    
    # 探索概率配置
    EXPLORATION_PROBABILITIES = {
        'low_memory_larger': 1.0,
        'low_memory_smaller': 0.2,
        'medium_memory_larger': 0.7,
        'medium_memory_smaller': 0.3,
        'high_memory_larger': 0.1,
        'high_memory_smaller': 1.0
    }
    
    # 评分权重配置
    SCORING_WEIGHTS = {
        'confidence_bonus_max': 0.1,
        'batch_bonus_low_memory': 0.2,
        'batch_bonus_medium_memory': 0.1,
        'batch_bonus_high_memory': 0.05
    }
    
    # 调整阈值
    ADJUSTMENT_THRESHOLDS = {
        'min_samples_required': 3,
        'score_improvement_threshold': 0.05,
        'recent_samples_keep': 20
    }


class DynamicBatchManager:
    """动态批处理管理器 - 优化版本"""
    
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
    
    def _get_memory_state(self) -> str:
        """获取显存状态分类"""
        usage = self.get_memory_usage()
        if usage < DynamicBatchConfig.MEMORY_THRESHOLDS['low']:
            return 'low'
        elif usage < DynamicBatchConfig.MEMORY_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'high'
    
    def _collect_performance_data(self) -> dict:
        """收集性能数据"""
        batch_performance = {}
        
        # 从持久化数据中获取性能信息
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
        
        return batch_performance
    
    def _filter_valid_batches(self, batch_performance: dict) -> dict:
        """过滤有效的批次数据"""
        min_samples = DynamicBatchConfig.ADJUSTMENT_THRESHOLDS['min_samples_required']
        return {bs: perf for bs, perf in batch_performance.items() 
                if len(perf['latencies']) >= min_samples}
    
    def _handle_insufficient_data(self):
        """处理数据不足的情况"""
        print(f"[DEBUG] Insufficient valid batch sizes for comparison")
        if len(self.batch_history) > 30:
            # 保留最近的样本
            recent_count = DynamicBatchConfig.ADJUSTMENT_THRESHOLDS['recent_samples_keep']
            self.batch_history = deque(list(self.batch_history)[-recent_count:], maxlen=self.adaptive_window)
            self.latency_history = deque(list(self.latency_history)[-recent_count:], maxlen=self.adaptive_window)
            self.throughput_history = deque(list(self.throughput_history)[-recent_count:], maxlen=self.adaptive_window)
            print(f"[INFO] Cleared old performance data, keeping recent {recent_count} samples")
    
    def _explore_new_batch_size(self, current_size: int):
        """探索新的批次大小"""
        memory_state = self._get_memory_state()
        config = DynamicBatchConfig.EXPLORATION_PROBABILITIES
        
        import random
        
        if memory_state == 'low':
            # 低显存压力，优先探索更大批次
            if current_size < self.max_batch_size and random.random() < config['low_memory_larger']:
                self.current_optimal_batch = min(self.max_batch_size, current_size + 1)
                print(f"[INFO] Low memory usage, exploring larger batch size -> {self.current_optimal_batch}")
            elif current_size > self.min_batch_size and random.random() < config['low_memory_smaller']:
                self.current_optimal_batch = max(self.min_batch_size, current_size - 1)
                print(f"[INFO] At max batch, trying smaller batch size -> {self.current_optimal_batch}")
        
        elif memory_state == 'medium':
            # 中等显存压力，平衡探索
            if current_size < self.max_batch_size and random.random() < config['medium_memory_larger']:
                self.current_optimal_batch = min(self.max_batch_size, current_size + 1)
                print(f"[INFO] Medium memory usage, exploring larger batch size -> {self.current_optimal_batch}")
            elif current_size > self.min_batch_size and random.random() < config['medium_memory_smaller']:
                self.current_optimal_batch = max(self.min_batch_size, current_size - 1)
                print(f"[INFO] Medium memory usage, exploring smaller batch size -> {self.current_optimal_batch}")
        
        else:  # high memory state
            # 高显存压力，优先探索更小批次
            if current_size > self.min_batch_size and random.random() < config['high_memory_smaller']:
                self.current_optimal_batch = max(self.min_batch_size, current_size - 1)
                print(f"[INFO] High memory usage, exploring smaller batch size -> {self.current_optimal_batch}")
            elif current_size < self.max_batch_size and random.random() < config['high_memory_larger']:
                self.current_optimal_batch = min(self.max_batch_size, current_size + 1)
                print(f"[INFO] At min batch, trying larger batch size -> {self.current_optimal_batch}")
    
    def _get_confidence_bonus(self, sample_count: int) -> float:
        """计算置信度奖励"""
        import math
        sample_weight = math.log(sample_count + 1) / math.log(10 + 1)
        return sample_weight * DynamicBatchConfig.SCORING_WEIGHTS['confidence_bonus_max']
    
    def _get_batch_bonus(self, batch_size: int, memory_state: str) -> float:
        """计算批次大小奖励"""
        weights = DynamicBatchConfig.SCORING_WEIGHTS
        
        if memory_state == 'low':
            return batch_size * weights['batch_bonus_low_memory']
        elif memory_state == 'medium':
            return batch_size * weights['batch_bonus_medium_memory']
        else:  # high
            return (self.max_batch_size - batch_size + 1) * weights['batch_bonus_high_memory']
    
    def _calculate_batch_score(self, batch_size: int, performance: dict, memory_state: str) -> float:
        """计算批次评分"""
        avg_latency = sum(performance['latencies']) / len(performance['latencies'])
        avg_throughput = sum(performance['throughputs']) / len(performance['throughputs'])
        sample_count = len(performance['latencies'])
        
        # 基础效率评分
        efficiency = avg_throughput / (1 + avg_latency / 50)
        
        # 置信度奖励
        confidence_bonus = self._get_confidence_bonus(sample_count)
        
        # 批次大小奖励
        batch_bonus = self._get_batch_bonus(batch_size, memory_state)
        
        # 综合评分
        score = efficiency + batch_bonus + confidence_bonus
        
        print(f"[DEBUG] Batch {batch_size}: efficiency={efficiency:.2f}, batch_bonus={batch_bonus:.2f}, confidence_bonus={confidence_bonus:.2f}, samples={sample_count}, total_score={score:.2f}")
        
        return score
    
    def _optimize_batch_size(self, valid_batches: dict):
        """优化批次大小"""
        memory_state = self._get_memory_state()
        best_score = -1
        best_batch_size = self.current_optimal_batch
        
        print(f"[DEBUG] Evaluating valid batch performance data: {[(k, len(v['latencies'])) for k, v in valid_batches.items()]}")
        
        # 计算所有有效批次的评分
        for batch_size, performance in valid_batches.items():
            score = self._calculate_batch_score(batch_size, performance, memory_state)
            
            if score > best_score:
                best_score = score
                best_batch_size = batch_size
        
        # 稳定的调整策略：只有显著改善才调整
        if best_batch_size != self.current_optimal_batch:
            current_score = self._calculate_batch_score(
                self.current_optimal_batch, 
                valid_batches.get(self.current_optimal_batch, {'latencies': [100], 'throughputs': [1]}),
                memory_state
            )
            
            score_improvement = (best_score - current_score) / max(current_score, 0.1)
            threshold = DynamicBatchConfig.ADJUSTMENT_THRESHOLDS['score_improvement_threshold']
            
            if score_improvement > threshold:
                # 渐进式调整：每次最多调整1个单位
                diff = best_batch_size - self.current_optimal_batch
                adjustment = 1 if diff > 0 else -1 if diff < 0 else 0
                
                self.current_optimal_batch = max(self.min_batch_size,
                                               min(self.max_batch_size,
                                                 self.current_optimal_batch + adjustment))
                
                print(f"[INFO] Significant improvement detected ({score_improvement:.1%}): optimal batch size -> {self.current_optimal_batch}")
            else:
                print(f"[DEBUG] Score improvement insufficient ({score_improvement:.1%}), keeping current optimal batch: {self.current_optimal_batch}")
    
    def _adjust_wait_time(self):
        """调整等待时间"""
        if len(self.latency_history) >= 10:
            avg_latency = sum(list(self.latency_history)[-10:]) / 10
            if avg_latency > 100:  # 延迟过高，减少等待时间
                self.current_wait_time_ms = max(10, self.current_wait_time_ms - 5)
            elif avg_latency < 50:  # 延迟较低，可以增加等待时间以获得更大批次
                self.current_wait_time_ms = min(self.max_wait_time_ms, 
                                              self.current_wait_time_ms + 5)
    
    def _adaptive_adjustment(self):
        """基于历史性能数据进行自适应调整 - 优化版本"""
        if len(self.batch_history) < 5:
            return
        
        batch_performance = self._collect_performance_data()
        valid_batches = self._filter_valid_batches(batch_performance)
        
        if len(valid_batches) < 2:
            self._handle_insufficient_data()
            return
        
        if len(valid_batches) == 1:
            self._explore_new_batch_size(list(valid_batches.keys())[0])
            return
        
        self._optimize_batch_size(valid_batches)
        self._adjust_wait_time()
    
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
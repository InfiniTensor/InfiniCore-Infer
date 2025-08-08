#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include "infinicore_infer.h"
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>

class AllocatorBase {
public:
    virtual void *alloc(size_t size) = 0;
    virtual void release(void *ptr) = 0;
    virtual ~AllocatorBase() = default;
};

// 内存使用统计结构
struct MemoryStats {
    std::atomic<size_t> total_allocated{0};
    std::atomic<size_t> total_freed{0};
    std::atomic<size_t> current_usage{0};
    std::atomic<size_t> peak_usage{0};
    std::atomic<size_t> allocation_count{0};
    std::atomic<size_t> free_count{0};
    std::atomic<size_t> fragmentation_events{0};
    
    void recordAllocation(size_t size) {
        total_allocated += size;
        current_usage += size;
        allocation_count++;
        
        size_t current = current_usage.load();
        size_t peak = peak_usage.load();
        while (current > peak && !peak_usage.compare_exchange_weak(peak, current)) {
            peak = peak_usage.load();
        }
    }
    
    void recordFree(size_t size) {
        total_freed += size;
        current_usage -= size;
        free_count++;
    }
    
    void recordFragmentation() {
        fragmentation_events++;
    }
    
    double getFragmentationRate() const {
        size_t allocs = allocation_count.load();
        return allocs > 0 ? static_cast<double>(fragmentation_events.load()) / allocs : 0.0;
    }
};

class MemoryPool : public AllocatorBase {
public:
    static constexpr size_t DEFAULT_ALIGNMENT = 256;
    static constexpr size_t SMALL_BLOCK_THRESHOLD = 1024;        // 1KB
    static constexpr size_t MEDIUM_BLOCK_THRESHOLD = 1024 * 1024; // 1MB
    static constexpr size_t LARGE_BLOCK_THRESHOLD = 16 * 1024 * 1024; // 16MB
    
    // 预分配配置
    struct PreallocationConfig {
        size_t small_pool_size;
        size_t medium_pool_size;
        size_t large_pool_size;
        bool enable_preallocation;
        
        PreallocationConfig() : 
            small_pool_size(16 * 1024 * 1024),
            medium_pool_size(128 * 1024 * 1024),
            large_pool_size(512 * 1024 * 1024),
            enable_preallocation(true) {}
    };

    explicit MemoryPool(size_t initialSize = 0, size_t alignment = DEFAULT_ALIGNMENT, 
                       const PreallocationConfig& config = PreallocationConfig{});
    ~MemoryPool();

    void *alloc(size_t size) override;
    void release(void *ptr) override;
    
    // 新增功能接口
    void defragment();  // 内存碎片整理
    const MemoryStats& getStats() const { return _stats; }
    void printStats() const;
    void preAllocate(const PreallocationConfig& config);  // 预分配内存
    bool shouldDefragment() const;  // 检查是否需要碎片整理
    
    size_t getAlignment() const { return _alignment; }
    size_t getTotalMemory() const;
    size_t getUsedMemory() const;
    size_t getFreeMemory() const;
    double getFragmentationRatio() const;

private:
    enum class BlockType {
        SMALL,
        MEDIUM, 
        LARGE
    };
    
    struct Block {
        void *base;
        void *ptr;
        size_t size;
        bool is_free;
        BlockType type;
        std::chrono::steady_clock::time_point last_used;

        Block(void *b, void *p, size_t s, bool f, BlockType t = BlockType::MEDIUM)
            : base(b), ptr(p), size(s), is_free(f), type(t), 
              last_used(std::chrono::steady_clock::now()) {}

        bool operator<(const Block &other) const {
            return ptr < other.ptr;
        }
    };
    
    struct PoolInfo {
        std::multimap<size_t, std::set<Block>::iterator> free_blocks;
        size_t total_size = 0;
        size_t used_size = 0;
    };

    BlockType getBlockType(size_t size) const;
    void *allocateNewRegion(size_t size, BlockType type = BlockType::MEDIUM);
    void tryCoalesce(const Block &block);
    void *allocFromPool(size_t size, BlockType type);
    void releaseToPool(void *ptr, const Block& block);
    
    // 碎片整理相关
    void compactPool(BlockType type);
    
    mutable std::mutex _mutex;  // 线程安全
    size_t _alignment;
    PreallocationConfig _config;
    
    std::vector<void *> _base_regions;
    std::set<Block> _all_blocks;
    std::unordered_map<void *, std::set<Block>::iterator> _ptr_to_block;
    
    // 分层内存管理
    PoolInfo _pools[3];  // SMALL, MEDIUM, LARGE
    
    // 统计信息
    mutable MemoryStats _stats;
    
    // 预分配的内存区域
    std::vector<void*> _preallocated_regions;
};

#endif

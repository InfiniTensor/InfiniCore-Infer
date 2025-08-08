#include "../allocator.hpp"
#include "../utils.hpp"
#include <iostream>
#include <algorithm>
#include <iomanip>

MemoryPool::MemoryPool(size_t initialSize, size_t alignment, const PreallocationConfig& config)
    : _alignment(alignment), _config(config) {
    // Validate alignment is power of two
    if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
        throw std::invalid_argument("Alignment must be a power of two");
    }

    // 预分配内存池
    if (_config.enable_preallocation) {
        preAllocate(_config);
    }
    
    if (initialSize > 0) {
        allocateNewRegion(initialSize);
    }
}

MemoryPool::~MemoryPool() {
    std::lock_guard<std::mutex> lock(_mutex);
    
    for (void *region : _base_regions) {
        RUN_INFINI(infinirtFree(region));
    }
    
    for (void *region : _preallocated_regions) {
        RUN_INFINI(infinirtFree(region));
    }
}

void *MemoryPool::alloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    
    // Calculate aligned size
    const size_t aligned_size = (size + _alignment - 1) & ~(_alignment - 1);
    
    // 确定块类型
    BlockType type = getBlockType(aligned_size);
    
    // 尝试从对应的池中分配
    void* ptr = allocFromPool(aligned_size, type);
    
    if (ptr) {
        _stats.recordAllocation(aligned_size);
        return ptr;
    }
    
    // 如果池中没有合适的块，分配新区域
    allocateNewRegion(std::max(aligned_size * 2, 
        type == BlockType::SMALL ? _config.small_pool_size / 4 :
        type == BlockType::MEDIUM ? _config.medium_pool_size / 4 :
        _config.large_pool_size / 4), type);
    
    ptr = allocFromPool(aligned_size, type);
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    _stats.recordAllocation(aligned_size);
    return ptr;
}

void MemoryPool::release(void *ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    
    auto it = _ptr_to_block.find(ptr);
    if (it == _ptr_to_block.end()) {
        throw std::runtime_error("Invalid pointer to free");
    }

    auto block_it = it->second;
    Block block = *block_it;
    
    _stats.recordFree(block.size);
    
    releaseToPool(ptr, block);
}

void *MemoryPool::allocateNewRegion(size_t size, BlockType type) {
    // Allocate exactly the requested size
    void *ptr = nullptr;
    RUN_INFINI(infinirtMalloc(&ptr, size));
    _base_regions.push_back(ptr);

    // Align the pointer within the allocated region
    size_t alignment_padding = 0; // ptr is already aligned from infinirtMalloc
    size_t usable_size = size - alignment_padding;

    Block new_block(ptr, ptr, usable_size, true, type);
    auto it = _all_blocks.insert(new_block).first;
    
    int pool_idx = static_cast<int>(type);
    _pools[pool_idx].free_blocks.emplace(usable_size, it);
    _pools[pool_idx].total_size += usable_size;

    return ptr;
}

MemoryPool::BlockType MemoryPool::getBlockType(size_t size) const {
    if (size <= SMALL_BLOCK_THRESHOLD) {
        return BlockType::SMALL;
    } else if (size <= MEDIUM_BLOCK_THRESHOLD) {
        return BlockType::MEDIUM;
    } else {
        return BlockType::LARGE;
    }
}

void *MemoryPool::allocFromPool(size_t size, BlockType type) {
    int pool_idx = static_cast<int>(type);
    auto& pool = _pools[pool_idx];
    
    // Find the first block with enough space
    auto it = pool.free_blocks.lower_bound(size);
    if (it == pool.free_blocks.end()) {
        return nullptr;
    }

    auto block_it = it->second;
    Block block = *block_it;
    pool.free_blocks.erase(it);
    _all_blocks.erase(block_it);

    // Calculate remaining space after allocation
    const size_t remaining = block.size - size;

    // Create allocated block
    Block alloc_block(block.base, block.ptr, size, false, type);
    auto alloc_it = _all_blocks.insert(alloc_block).first;
    _ptr_to_block[block.ptr] = alloc_it;
    
    pool.used_size += size;

    // Split remaining space if it's large enough
    if (remaining >= _alignment) {
        void *rem_ptr = static_cast<char *>(block.ptr) + size;
        Block rem_block(block.base, rem_ptr, remaining, true, type);
        auto rem_it = _all_blocks.insert(rem_block).first;
        pool.free_blocks.emplace(remaining, rem_it);
    }

    return block.ptr;
}

void MemoryPool::releaseToPool(void *ptr, const Block& block) {
    _all_blocks.erase(_ptr_to_block[ptr]);
    _ptr_to_block.erase(ptr);
    
    Block free_block = block;
    free_block.is_free = true;
    free_block.last_used = std::chrono::steady_clock::now();
    
    auto new_it = _all_blocks.insert(free_block).first;
    
    int pool_idx = static_cast<int>(block.type);
    _pools[pool_idx].used_size -= block.size;
    
    tryCoalesce(*new_it);
}

void MemoryPool::tryCoalesce(const Block &block) {
    auto it = _all_blocks.find(block);
    if (it == _all_blocks.end()) {
        return;
    }

    Block merged = *it;
    auto next = std::next(it);
    auto prev = (it == _all_blocks.begin()) ? _all_blocks.end() : std::prev(it);

    _all_blocks.erase(it);
    
    int pool_idx = static_cast<int>(merged.type);
    auto& pool = _pools[pool_idx];
    
    // Remove from free blocks
    auto range = pool.free_blocks.equal_range(merged.size);
    for (auto free_it = range.first; free_it != range.second; ++free_it) {
        if (free_it->second->ptr == merged.ptr) {
            pool.free_blocks.erase(free_it);
            break;
        }
    }

    // Coalesce with next
    if (next != _all_blocks.end() && next->is_free && 
        next->type == merged.type &&
        static_cast<char *>(merged.ptr) + merged.size == next->ptr) {
        
        // Remove next from free blocks
        auto next_range = pool.free_blocks.equal_range(next->size);
        for (auto free_it = next_range.first; free_it != next_range.second; ++free_it) {
            if (free_it->second->ptr == next->ptr) {
                pool.free_blocks.erase(free_it);
                break;
            }
        }
        
        merged.size += next->size;
        _all_blocks.erase(next);
        _stats.recordFragmentation();
    }

    // Coalesce with prev
    if (prev != _all_blocks.end() && prev->is_free && 
        prev->type == merged.type &&
        static_cast<char *>(prev->ptr) + prev->size == merged.ptr) {
        
        // Remove prev from free blocks
        auto prev_range = pool.free_blocks.equal_range(prev->size);
        for (auto free_it = prev_range.first; free_it != prev_range.second; ++free_it) {
            if (free_it->second->ptr == prev->ptr) {
                pool.free_blocks.erase(free_it);
                break;
            }
        }
        
        merged.ptr = prev->ptr;
        merged.size += prev->size;
        merged.base = prev->base;
        _all_blocks.erase(prev);
        _stats.recordFragmentation();
    }

    merged.is_free = true;
    auto new_it = _all_blocks.insert(merged).first;
    pool.free_blocks.emplace(merged.size, new_it);
}

void MemoryPool::preAllocate(const PreallocationConfig& config) {
    if (config.small_pool_size > 0) {
        void* ptr = nullptr;
        RUN_INFINI(infinirtMalloc(&ptr, config.small_pool_size));
        _preallocated_regions.push_back(ptr);
        
        Block small_block(ptr, ptr, config.small_pool_size, true, BlockType::SMALL);
        auto it = _all_blocks.insert(small_block).first;
        _pools[0].free_blocks.emplace(config.small_pool_size, it);
        _pools[0].total_size += config.small_pool_size;
    }
    
    if (config.medium_pool_size > 0) {
        void* ptr = nullptr;
        RUN_INFINI(infinirtMalloc(&ptr, config.medium_pool_size));
        _preallocated_regions.push_back(ptr);
        
        Block medium_block(ptr, ptr, config.medium_pool_size, true, BlockType::MEDIUM);
        auto it = _all_blocks.insert(medium_block).first;
        _pools[1].free_blocks.emplace(config.medium_pool_size, it);
        _pools[1].total_size += config.medium_pool_size;
    }
    
    if (config.large_pool_size > 0) {
        void* ptr = nullptr;
        RUN_INFINI(infinirtMalloc(&ptr, config.large_pool_size));
        _preallocated_regions.push_back(ptr);
        
        Block large_block(ptr, ptr, config.large_pool_size, true, BlockType::LARGE);
        auto it = _all_blocks.insert(large_block).first;
        _pools[2].free_blocks.emplace(config.large_pool_size, it);
        _pools[2].total_size += config.large_pool_size;
    }
}

void MemoryPool::defragment() {
    std::lock_guard<std::mutex> lock(_mutex);
    
    for (int i = 0; i < 3; ++i) {
        compactPool(static_cast<BlockType>(i));
    }
}

void MemoryPool::compactPool(BlockType type) {
    int pool_idx = static_cast<int>(type);
    auto& pool = _pools[pool_idx];
    
    // 收集所有空闲块
    std::vector<std::set<Block>::iterator> free_blocks;
    for (auto& pair : pool.free_blocks) {
        free_blocks.push_back(pair.second);
    }
    
    // 按地址排序
    std::sort(free_blocks.begin(), free_blocks.end(), 
              [](const auto& a, const auto& b) {
                  return a->ptr < b->ptr;
              });
    
    // 尝试合并相邻的空闲块
    for (size_t i = 0; i < free_blocks.size(); ++i) {
        auto current = free_blocks[i];
        if (current == _all_blocks.end()) continue;
        
        for (size_t j = i + 1; j < free_blocks.size(); ++j) {
            auto next = free_blocks[j];
            if (next == _all_blocks.end()) continue;
            
            if (static_cast<char*>(current->ptr) + current->size == next->ptr) {
                // 合并块
                Block merged = *current;
                merged.size += next->size;
                
                // 从池中移除旧块
                pool.free_blocks.erase(current->size);
                pool.free_blocks.erase(next->size);
                
                _all_blocks.erase(current);
                _all_blocks.erase(next);
                
                // 插入新的合并块
                auto new_it = _all_blocks.insert(merged).first;
                pool.free_blocks.emplace(merged.size, new_it);
                
                free_blocks[i] = new_it;
                free_blocks[j] = _all_blocks.end();
                
                current = new_it;
            }
        }
    }
}

bool MemoryPool::shouldDefragment() const {
    return _stats.getFragmentationRate() > 0.3; // 30%碎片率阈值
}

void MemoryPool::printStats() const {
    std::cout << "\n=== Memory Pool Statistics ===\n";
    std::cout << "Total Allocated: " << _stats.total_allocated.load() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Total Freed: " << _stats.total_freed.load() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Current Usage: " << _stats.current_usage.load() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Peak Usage: " << _stats.peak_usage.load() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Allocation Count: " << _stats.allocation_count.load() << "\n";
    std::cout << "Free Count: " << _stats.free_count.load() << "\n";
    std::cout << "Fragmentation Rate: " << std::fixed << std::setprecision(2) 
              << _stats.getFragmentationRate() * 100 << "%\n";
    
    std::cout << "\n=== Pool Information ===\n";
    const char* pool_names[] = {"Small", "Medium", "Large"};
    for (int i = 0; i < 3; ++i) {
        std::cout << pool_names[i] << " Pool:\n";
        std::cout << "  Total Size: " << _pools[i].total_size / (1024.0 * 1024.0) << " MB\n";
        std::cout << "  Used Size: " << _pools[i].used_size / (1024.0 * 1024.0) << " MB\n";
        std::cout << "  Free Blocks: " << _pools[i].free_blocks.size() << "\n";
        std::cout << "  Utilization: " << std::fixed << std::setprecision(2)
                  << (_pools[i].total_size > 0 ? 
                      static_cast<double>(_pools[i].used_size) / _pools[i].total_size * 100 : 0) 
                  << "%\n\n";
    }
}

size_t MemoryPool::getTotalMemory() const {
    std::lock_guard<std::mutex> lock(_mutex);
    size_t total = 0;
    for (int i = 0; i < 3; ++i) {
        total += _pools[i].total_size;
    }
    return total;
}

size_t MemoryPool::getUsedMemory() const {
    return _stats.current_usage.load();
}

size_t MemoryPool::getFreeMemory() const {
    return getTotalMemory() - getUsedMemory();
}

double MemoryPool::getFragmentationRatio() const {
    return _stats.getFragmentationRate();
}

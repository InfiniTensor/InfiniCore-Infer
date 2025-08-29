#ifndef INFER_TENSOR_H
#define INFER_TENSOR_H

#include "allocator.hpp"
#include "utils.hpp"
#include <memory>
#include <string>
#include <vector>

class Storage {
private:
    Storage() = default;
    void *_memory;
    size_t _size;
    infiniDevice_t _device_type;
    int _device_id;
    std::shared_ptr<MemoryPool> _memory_pool;

public:
    static std::shared_ptr<Storage> create(size_t size);
    static std::shared_ptr<Storage> createAsync(size_t size, infinirtStream_t stream = nullptr);
    static std::shared_ptr<Storage> createFromPool(size_t size, std::shared_ptr<MemoryPool> pool = nullptr);
    static std::shared_ptr<Storage> createHost(size_t size);
    ~Storage();

    void *memory() const { return _memory; }
    size_t size() const { return _size; }
    infiniDevice_t deviceType() const { return _device_type; }
    int deviceId() const { return _device_id; }
};

struct SliceParams {
    size_t dim;
    size_t start;
    size_t len;
};

template <typename... Args>
std::vector<size_t> __shape(Args... args) {
    return std::vector<size_t>{static_cast<size_t>(args)...};
}

template <typename... Args>
std::vector<ptrdiff_t> __strides(Args... args) {
    return std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(args)...};
}
class TensorDesc {
private:
    infiniDtype_t _dtype;
    std::vector<size_t> _shape;
    std::vector<ptrdiff_t> _strides;
    infiniopTensorDescriptor_t _desc;
    size_t _seed;

    TensorDesc(infiniDtype_t dtype, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides) : _dtype(dtype), _shape(shape), _strides(strides), _desc(nullptr) { computeTensorDesHash(); }
    void resetDesc();
    void computeTensorDesHash();

public:
    ~TensorDesc();
    static std::shared_ptr<TensorDesc>
    create(infiniDtype_t dtype, const std::vector<size_t> &shape,
           const std::vector<ptrdiff_t> &strides);
    static std::shared_ptr<TensorDesc>
    create(infiniDtype_t dtype, const std::vector<size_t> &shape);
    static std::shared_ptr<TensorDesc>
    createWithOrder(infiniDtype_t dtype, const std::vector<size_t> &shape,
                    const std::vector<size_t> &order);

    infiniDtype_t dtype() const { return _dtype; }
    const std::vector<size_t> &shape() const { return _shape; }
    const std::vector<ptrdiff_t> &strides() const { return _strides; }
    size_t ndim() const { return _shape.size(); }
    infiniopTensorDescriptor_t desc() const;
    bool isContigous() const;
    std::string info() const;
    size_t seed() const { return _seed; }

    void dimMerge(size_t dim_start, size_t dim_end);
    void dimSplit(size_t dim, const std::vector<size_t> &dims);
    void permute(const std::vector<size_t> &order);
};

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::shared_ptr<Storage> _storage;
    std::shared_ptr<const TensorDesc> _desc;

    ptrdiff_t _offset;

    void *dataImpl(ptrdiff_t offset) const;
    std::shared_ptr<Tensor>
    sliceImpl(const std::vector<SliceParams> &slices) const;

public:
    static std::shared_ptr<Tensor> buffer(infiniDtype_t dtype,
                                          const std::vector<size_t> &shape,
                                          std::shared_ptr<MemoryPool> pool = nullptr);
    static std::shared_ptr<Tensor> weight(void *host_data,
                                          infiniDtype_t dtype,
                                          const std::vector<size_t> &shape);
    void load(const void *host_data, infinirtStream_t stream = nullptr);
    std::shared_ptr<Tensor> memShare(const std::vector<size_t> &shape,
                                     infiniDtype_t dtype = INFINI_DTYPE_INVALID) const;
    std::shared_ptr<Tensor> slice(size_t dim, size_t start, size_t len);
    std::shared_ptr<Tensor const> slice(size_t dim, size_t start,
                                        size_t len) const;
    std::shared_ptr<Tensor> slice(const std::vector<SliceParams> &slices);
    std::shared_ptr<Tensor const>
    slice(const std::vector<SliceParams> &slices) const;
    std::shared_ptr<Tensor> dimMerge(size_t dim_start, size_t dim_end);
    std::shared_ptr<Tensor> dimSplit(size_t dim,
                                     const std::vector<size_t> &dims);
    std::shared_ptr<Tensor> permute(const std::vector<size_t> &order);
    void *data(ptrdiff_t offset = 0);
    void const *data(ptrdiff_t offset = 0) const;
    void copyFrom(std::shared_ptr<Tensor const> src, infiniopHandle_t handle,
                  infinirtStream_t stream = nullptr);
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    size_t ndim() const;
    infiniDtype_t dtype() const;
    bool isContigous() const;
    infiniopTensorDescriptor_t desc() const;
    ptrdiff_t dataOffset() const;
    infiniDevice_t deviceType() const;
    int deviceId() const;
    size_t numel() const;

    void debug(const std::string &filename) const;
    void debug() const;
    std::string info() const;
    size_t seed() const;

    std::shared_ptr<Tensor> view(const std::vector<size_t> &new_shape) const;
    std::shared_ptr<Tensor> view_as(const std::vector<size_t> &new_shape) const;
    std::shared_ptr<Tensor> view_as(const std::vector<size_t> &new_shape, const std::vector<ptrdiff_t> &new_strides) const;

    ~Tensor();
};

inline size_t dsize(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_INVALID:
        return 0;
    case INFINI_DTYPE_BYTE:
        return 1;
    case INFINI_DTYPE_BOOL:
        return 1;
    case INFINI_DTYPE_I8:
        return 1;
    case INFINI_DTYPE_I16:
        return 2;
    case INFINI_DTYPE_I32:
        return 4;
    case INFINI_DTYPE_I64:
        return 8;
    case INFINI_DTYPE_U8:
        return 1;
    case INFINI_DTYPE_U16:
        return 2;
    case INFINI_DTYPE_U32:
        return 4;
    case INFINI_DTYPE_U64:
        return 8;
    case INFINI_DTYPE_F8:
        return 1;
    case INFINI_DTYPE_F16:
        return 2;
    case INFINI_DTYPE_F32:
        return 4;
    case INFINI_DTYPE_F64:
        return 8;
    case INFINI_DTYPE_C16:
        return 2;
    case INFINI_DTYPE_C32:
        return 4;
    case INFINI_DTYPE_C64:
        return 8;
    case INFINI_DTYPE_C128:
        return 16;
    case INFINI_DTYPE_BF16:
        return 2;
    default:
        return 0;
    }
}

#endif

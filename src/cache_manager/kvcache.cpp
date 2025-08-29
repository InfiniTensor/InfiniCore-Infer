#include "../cache.hpp"

__C struct KVCache *createKVCache(
    size_t nlayers,
    size_t max_len,
    size_t nkvh_,
    size_t dk,
    size_t dv,
    infiniDtype_t dtype,
    infiniDevice_t device,
    int *dev_ids,
    size_t ndev) {

    KVCache *cache = new KVCache();
    auto nkvh = nkvh_ / ndev;

    auto shape_k = std::vector<size_t>{max_len, nkvh, dk};
    auto shape_v = std::vector<size_t>{max_len, nkvh, dv};
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(device, dev_ids[idev]));
        auto kcache = std::vector<std::shared_ptr<Tensor>>();
        auto vcache = std::vector<std::shared_ptr<Tensor>>();
        for (unsigned int layer = 0; layer < nlayers; layer++) {
            kcache.push_back(std::move(Tensor::buffer(dtype, shape_k)));
            vcache.push_back(std::move(Tensor::buffer(dtype, shape_v)));
        }
        cache->k.push_back(kcache);
        cache->v.push_back(vcache);
    }

    return cache;
}

__C struct KVCache *duplicateKVCache(const KVCache *kv_cache, size_t seq_len) {
    auto ndev = kv_cache->k.size();
    auto nlayers = kv_cache->k[0].size();
    auto device = kv_cache->k[0][0]->deviceType();
    auto dtype = kv_cache->k[0][0]->dtype();
    auto shape_k = kv_cache->k[0][0]->shape();
    auto shape_v = kv_cache->v[0][0]->shape();
    auto size_k = seq_len * shape_k[1] * shape_k[2] * dsize(dtype);
    auto size_v = seq_len * shape_v[1] * shape_v[2] * dsize(dtype);
    KVCache *new_kv_cache = new KVCache();
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(device, kv_cache->k[idev][0]->deviceId()));
        for (unsigned int layer = 0; layer < nlayers; layer++) {
            auto kcache = std::vector<std::shared_ptr<Tensor>>();
            auto vcache = std::vector<std::shared_ptr<Tensor>>();
            for (unsigned int layer = 0; layer < nlayers; layer++) {
                kcache.push_back(std::move(Tensor::buffer(dtype, shape_k)));
                vcache.push_back(std::move(Tensor::buffer(dtype, shape_v)));
            }
            new_kv_cache->k.push_back(kcache);
            new_kv_cache->v.push_back(vcache);
            RUN_INFINI(infinirtMemcpy(new_kv_cache->k[idev][layer]->data(),
                                      kv_cache->k[idev][layer]->data(),
                                      size_k,
                                      INFINIRT_MEMCPY_D2D));
            RUN_INFINI(infinirtMemcpy(new_kv_cache->v[idev][layer]->data(),
                                      kv_cache->v[idev][layer]->data(),
                                      size_v,
                                      INFINIRT_MEMCPY_D2D));
        }
    }
    return new_kv_cache;
}

__C void dropKVCache(KVCache *kv_cache) {
    auto ndev = kv_cache->k.size();
    auto nlayers = kv_cache->k[0].size();
    auto device = kv_cache->k[0][0]->deviceType();
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(device, kv_cache->k[idev][0]->deviceId()));
        for (unsigned int layer = 0; layer < nlayers; layer++) {
            kv_cache->k[idev][layer].reset();
            kv_cache->v[idev][layer].reset();
        }
    }
    delete kv_cache;
}

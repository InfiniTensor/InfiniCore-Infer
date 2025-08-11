#include "mixtral_impl.hpp"

__C struct KVCache *createMixtralKVCache(const MixtralModel *model) {
    KVCache *cache = new KVCache();
    auto ndev = model->dev_resources.size();
    auto nkvh = model->meta.nkvh / ndev;
    auto max_len = model->meta.dctx;
    auto dh = model->meta.d / model->meta.nh;
    auto shape = std::vector<size_t>{(size_t)max_len, (size_t)nkvh, (size_t)dh};
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        auto kcache = std::vector<std::shared_ptr<Tensor>>();
        auto vcache = std::vector<std::shared_ptr<Tensor>>();
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
            vcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
        }
        cache->k.push_back(kcache);
        cache->v.push_back(vcache);
    }

    return cache;
}

__C struct KVCache *duplicateMixtralKVCache(const MixtralModel *model,
                                     const KVCache *kv_cache,
                                     unsigned int seq_len) {
    auto new_kv_cache = createMixtralKVCache(model);
    auto ndev = model->dev_resources.size();
    auto nkvh = model->meta.nkvh / ndev;
    auto dh = model->meta.d / model->meta.nh;
    auto dt_size = dsize(model->meta.dt_logits);
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            RUN_INFINI(infinirtMemcpy(new_kv_cache->k[idev][layer]->data(),
                                      kv_cache->k[idev][layer]->data(),
                                      seq_len * nkvh * dh * dt_size,
                                      INFINIRT_MEMCPY_D2D));
            RUN_INFINI(infinirtMemcpy(new_kv_cache->v[idev][layer]->data(),
                                      kv_cache->v[idev][layer]->data(),
                                      seq_len * nkvh * dh * dt_size,
                                      INFINIRT_MEMCPY_D2D));
        }
    }
    return new_kv_cache;
}

__C void dropMixtralKVCache(MixtralModel const *model, KVCache *kv_cache) {
    auto ndev = model->dev_resources.size();
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kv_cache->k[idev][layer].reset();
            kv_cache->v[idev][layer].reset();
        }
    }
    delete kv_cache;
}
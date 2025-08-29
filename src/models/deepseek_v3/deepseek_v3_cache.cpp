#include "deepseek_v3_impl.hpp"

__C struct DeepSeekV3Cache *
createDeepSeekV3Cache(const struct DeepSeekV3Model *model) {
    DeepSeekV3Cache *cache = new DeepSeekV3Cache();
    auto ndev = model->dev_resources.size();
    auto nlayer = model->meta.n_dense_layer + model->meta.n_sparse_layer;
    auto max_len = model->meta.dctx;
    auto d_rope = model->meta.d_rope;
    auto r_kv = model->meta.r_kv;
    auto kv_pass_shape = std::vector<size_t>{max_len, r_kv};
    auto k_rot_shape = std::vector<size_t>{max_len, d_rope};
    for (size_t idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        auto kv_pass_cache = std::vector<std::shared_ptr<Tensor>>();
        auto k_rot_cache = std::vector<std::shared_ptr<Tensor>>();
        for (size_t layer = 0; layer < nlayer; layer++) {
            kv_pass_cache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, kv_pass_shape)));
            k_rot_cache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, k_rot_shape)));
        }
        cache->kv_pass.push_back(kv_pass_cache);
        cache->k_rot.push_back(k_rot_cache);
    }

    return cache;
}

__C void
dropDeepSeekV3Cache(const struct DeepSeekV3Model *model,
                    struct DeepSeekV3Cache *cache) {
    auto ndev = model->dev_resources.size();
    auto nlayer = model->meta.n_dense_layer + model->meta.n_sparse_layer;
    for (size_t idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (size_t layer = 0; layer < nlayer; layer++) {
            cache->kv_pass[idev][layer].reset();
            cache->k_rot[idev][layer].reset();
        }
    }
    delete cache;
}
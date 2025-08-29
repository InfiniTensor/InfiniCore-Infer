#include "deepseek_v3_impl.hpp"

#include <cmath>

inline std::shared_ptr<Tensor> getInEmbd(
    const DeepSeekV3Meta *meta) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight(nullptr, meta->dt_logits, shape);
}

inline std::shared_ptr<Tensor> getOutNorm(
    const DeepSeekV3Meta *meta) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight(nullptr, meta->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getOutEmbd(
    const DeepSeekV3Meta *meta) {

    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight(nullptr, meta->dt_logits, shape)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> getMLANorm(
    const DeepSeekV3Meta *meta) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight(nullptr, meta->dt_norm, shape);
}

inline std::shared_ptr<QuantLinearWeight> getQuantLinear(
    const DeepSeekV3Meta *meta, size_t in_dim, size_t out_dim) {
    auto qw = std::make_shared<QuantLinearWeight>();
    auto shape_w = std::vector<size_t>({in_dim, out_dim / 8});
    qw->w = Tensor::weight(nullptr, INFINI_DTYPE_I32, shape_w);
    qw->s = Tensor::weight(nullptr, meta->dt_quant_scale, {in_dim / 64, out_dim});
    qw->z = Tensor::weight(nullptr, INFINI_DTYPE_I32, {in_dim / 64, out_dim / 8});
    return qw;
}

// ------------------- MLA Weights -------------------
inline std::shared_ptr<Tensor> getMLPNorm(
    const DeepSeekV3Meta *meta) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight(nullptr, meta->dt_norm, shape);
}

inline std::shared_ptr<MLAWeight> getMLA(const DeepSeekV3Meta *meta, int ndev) {
    auto mla = std::make_shared<MLAWeight>();

    mla->q_a_proj = getQuantLinear(meta, meta->d, meta->r_q);
    mla->q_a_norm = Tensor::weight(nullptr, meta->dt_norm, {meta->r_q});
    mla->q_b_proj = getQuantLinear(meta, meta->r_q, meta->nh / ndev * meta->d_qk);

    mla->kv_a_proj = getQuantLinear(meta, meta->d, meta->r_kv + meta->d_rope);
    mla->kv_a_norm = Tensor::weight(nullptr, meta->dt_norm, {meta->r_kv});
    mla->kv_b_proj = getQuantLinear(meta, meta->r_kv, meta->nh / ndev * (meta->d_nope + meta->d_v));

    mla->o_proj = getQuantLinear(meta, meta->nh / ndev * meta->d_v, meta->d);
    return mla;
}

// ------------------- Dense MLP -------------------

inline std::shared_ptr<MLPWeight> getMLP(const DeepSeekV3Meta *meta, size_t d, size_t di) {
    auto mlp = std::make_shared<MLPWeight>();
    mlp->gate = getQuantLinear(meta, d, di);
    mlp->up = getQuantLinear(meta, d, di);
    mlp->down = getQuantLinear(meta, di, d);
    return mlp;
}

inline std::shared_ptr<MLPWeight> getDenseMLP(const DeepSeekV3Meta *meta, int ndev) {
    return getMLP(meta, meta->d, meta->di / ndev);
}

// ------------------- Sparse Route + Experts -------------------

inline std::shared_ptr<GateWeight> getRouteWeight(
    const DeepSeekV3Meta *meta) {
    auto gw = std::make_shared<GateWeight>();
    gw->w = Tensor::weight(nullptr, meta->dt_gate_weight, {meta->nexperts, meta->d})->permute({1, 0});
    gw->b = Tensor::weight(nullptr, meta->dt_gate_bias, {meta->nexperts});
    return gw;
}

inline std::shared_ptr<MLPWeight> getShareExpert(const DeepSeekV3Meta *meta, int ndev) {
    return getMLP(meta, meta->d, meta->di_moe / ndev);
}

inline std::vector<std::shared_ptr<MLPWeight>> getExperts(const DeepSeekV3Meta *meta, int ndev) {
    std::vector<std::shared_ptr<MLPWeight>> experts(meta->nexperts);
    for (size_t i = 0; i < meta->nexperts; i++) {
        experts[i] = getMLP(meta, meta->d, meta->di_moe / ndev);
    }
    return experts;
}

inline std::shared_ptr<Tensor> getSinTable(const DeepSeekV3Meta *meta) {
    auto half_dh = meta->d_rope / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->rope_theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(const DeepSeekV3Meta *meta) {
    auto half_dh = meta->d_rope / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->rope_theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

DeepSeekV3Weights::DeepSeekV3Weights(
    const DeepSeekV3Meta *meta, infiniDevice_t device, int ndev, const int *dev_ids) {
    device_weights = std::vector<std::shared_ptr<DeepSeekV3DeviceWeights>>(ndev);
    for (int dev = 0; dev < ndev; dev++) {
        int dev_id = dev_ids[dev];
        RUN_INFINI(infinirtSetDevice(device, dev_id));
        device_weights[dev] = std::make_shared<DeepSeekV3DeviceWeights>();
        device_weights[dev]->device = device;
        device_weights[dev]->dev_id = dev_id;
        RUN_INFINI(infinirtStreamCreate(&device_weights[dev]->load_stream));

        device_weights[dev]->w_in_embd = getInEmbd(meta);
        device_weights[dev]->w_out_norm = getOutNorm(meta);
        device_weights[dev]->w_out_embd = getOutEmbd(meta);
        device_weights[dev]->sin_table = getSinTable(meta);
        device_weights[dev]->cos_table = getCosTable(meta);

        device_weights[dev]->w_layers = std::vector<LayerWeight>(meta->n_dense_layer + meta->n_sparse_layer);

        for (size_t layer = 0; layer < meta->n_dense_layer + meta->n_sparse_layer; layer++) {
            device_weights[dev]->w_layers[layer].mla_norm = getMLANorm(meta);
            device_weights[dev]->w_layers[layer].mla = getMLA(meta, ndev);
            device_weights[dev]->w_layers[layer].mlp_norm = getMLPNorm(meta);
            if (layer < meta->n_dense_layer) {
                device_weights[dev]->w_layers[layer].dense_mlp = getDenseMLP(meta, ndev);
            } else {
                device_weights[dev]->w_layers[layer].route = getRouteWeight(meta);
                device_weights[dev]->w_layers[layer].share_expert = getShareExpert(meta, ndev);
                device_weights[dev]->w_layers[layer].experts = getExperts(meta, ndev);
            }
        }
    }
}

// --- Global
void load_input_embd(DeepSeekV3Weights *weights, void *cpu_ptr) {
    std::cout << "Loading input embedding from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_in_embd->load(cpu_ptr, weight->load_stream);
    }
}

void load_output_norm(DeepSeekV3Weights *weights, void *cpu_ptr) {
    std::cout << "Loading output norm from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_out_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_output_embd(DeepSeekV3Weights *weights, void *cpu_ptr) {
    std::cout << "Loading output embedding from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_out_embd->load(cpu_ptr, weight->load_stream);
    }
}

// --- Attention
void load_attn_norm(DeepSeekV3Weights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading attention norm " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer].mla_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_attn_q_a_proj(DeepSeekV3Weights *weights,
                        void *weight_ptr, void *scale_ptr, void *zero_ptr, size_t layer) {
    std::cout << "Loading attention q_a_proj " << layer << " from " << weight_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer].mla->q_a_proj->w->load(weight_ptr, weight->load_stream);
        weight->w_layers[layer].mla->q_a_proj->s->load(scale_ptr, weight->load_stream);
        weight->w_layers[layer].mla->q_a_proj->z->load(zero_ptr, weight->load_stream);
    }
}

void load_attn_q_a_layernorm(DeepSeekV3Weights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading attention q_a_layernorm " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer].mla->q_a_norm->load(cpu_ptr, weight->load_stream);
    }
}

inline void load_dist_linear(void *w_ptr, void *s_ptr, void *z_ptr, std::shared_ptr<Tensor> w, std::shared_ptr<Tensor> s, std::shared_ptr<Tensor> z, size_t ndev, size_t dev, infinirtStream_t stream) {
    auto w_offset = w->shape()[0] * w->shape()[1] / ndev * dev * dsize(w->dtype());
    auto s_offset = s->shape()[0] * s->shape()[1] / ndev * dev * dsize(s->dtype());
    auto z_offset = z->shape()[0] * z->shape()[1] / ndev * dev * dsize(z->dtype());
    w->load(reinterpret_cast<char *>(w_ptr) + w_offset, stream);
    s->load(reinterpret_cast<char *>(s_ptr) + s_offset, stream);
    z->load(reinterpret_cast<char *>(z_ptr) + z_offset, stream);
}

void load_attn_q_b_proj(DeepSeekV3Weights *weights,
                        void *weight_ptr, void *scale_ptr, void *zero_ptr, size_t layer) {
    std::cout << "Loading attention q_b_proj " << layer << " from " << weight_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        auto w = weight->w_layers[layer].mla->q_b_proj->w;
        auto s = weight->w_layers[layer].mla->q_b_proj->s;
        auto z = weight->w_layers[layer].mla->q_b_proj->z;
        load_dist_linear(weight_ptr, scale_ptr, zero_ptr, w, s, z, weights->device_weights.size(), dev, weight->load_stream);
    }
}

void load_attn_kv_a_proj_with_mqa(DeepSeekV3Weights *weights,
                                  void *weight_ptr, void *scale_ptr, void *zero_ptr, size_t layer) {
    std::cout << "Loading attention kv_a_proj_with_mqa " << layer << " from " << weight_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer].mla->kv_a_proj->w->load(weight_ptr, weight->load_stream);
        weight->w_layers[layer].mla->kv_a_proj->s->load(scale_ptr, weight->load_stream);
        weight->w_layers[layer].mla->kv_a_proj->z->load(zero_ptr, weight->load_stream);
    }
}

void load_attn_kv_a_layernorm(DeepSeekV3Weights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading attention kv_a_layernorm " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer].mla->kv_a_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_attn_kv_b_proj(DeepSeekV3Weights *weights,
                         void *weight_ptr, void *scale_ptr, void *zero_ptr, size_t layer) {
    std::cout << "Loading attention kv_b_proj " << layer << " from " << weight_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        auto w = weight->w_layers[layer].mla->kv_b_proj->w;
        auto s = weight->w_layers[layer].mla->kv_b_proj->s;
        auto z = weight->w_layers[layer].mla->kv_b_proj->z;
        load_dist_linear(weight_ptr, scale_ptr, zero_ptr, w, s, z, weights->device_weights.size(), dev, weight->load_stream);
    }
}

void load_attn_o_proj(DeepSeekV3Weights *weights,
                      void *weight_ptr, void *scale_ptr, void *zero_ptr, size_t layer) {
    std::cout << "Loading attention o_proj " << layer << " from " << weight_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        auto w = weight->w_layers[layer].mla->o_proj->w;
        auto s = weight->w_layers[layer].mla->o_proj->s;
        auto z = weight->w_layers[layer].mla->o_proj->z;
        load_dist_linear(weight_ptr, scale_ptr, zero_ptr, w, s, z, weights->device_weights.size(), dev, weight->load_stream);
    }
}

// --- MLP
void load_mlp_norm(DeepSeekV3Weights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading mlp norm " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer].mlp_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_mlp_dense(DeepSeekV3Weights *weights,
                    void *gate_weight_ptr, void *gate_scale_ptr, void *gate_zero_ptr,
                    void *up_weight_ptr, void *up_scale_ptr, void *up_zero_ptr,
                    void *down_weight_ptr, void *down_scale_ptr, void *down_zero_ptr,
                    size_t layer_id) {
    std::cout << "Loading mlp dense " << layer_id << " from " << gate_weight_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        auto gate_w = weight->w_layers[layer_id].dense_mlp->gate->w;
        auto gate_s = weight->w_layers[layer_id].dense_mlp->gate->s;
        auto gate_z = weight->w_layers[layer_id].dense_mlp->gate->z;
        auto up_w = weight->w_layers[layer_id].dense_mlp->up->w;
        auto up_s = weight->w_layers[layer_id].dense_mlp->up->s;
        auto up_z = weight->w_layers[layer_id].dense_mlp->up->z;
        auto down_w = weight->w_layers[layer_id].dense_mlp->down->w;
        auto down_s = weight->w_layers[layer_id].dense_mlp->down->s;
        auto down_z = weight->w_layers[layer_id].dense_mlp->down->z;
        load_dist_linear(gate_weight_ptr, gate_scale_ptr, gate_zero_ptr, gate_w, gate_s, gate_z, weights->device_weights.size(), dev, weight->load_stream);
        load_dist_linear(up_weight_ptr, up_scale_ptr, up_zero_ptr, up_w, up_s, up_z, weights->device_weights.size(), dev, weight->load_stream);
        load_dist_linear(down_weight_ptr, down_scale_ptr, down_zero_ptr, down_w, down_s, down_z, weights->device_weights.size(), dev, weight->load_stream);
    }
}

void load_mlp_gate_weight(DeepSeekV3Weights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading mlp gate weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer].route->w->load(cpu_ptr, weight->load_stream);
    }
}

void load_mlp_gate_bias(DeepSeekV3Weights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading mlp gate bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer].route->b->load(cpu_ptr, weight->load_stream);
    }
}

void load_mlp_shared_experts(DeepSeekV3Weights *weights,
                             void *gate_weight_ptr, void *gate_scale_ptr, void *gate_zero_ptr,
                             void *up_weight_ptr, void *up_scale_ptr, void *up_zero_ptr,
                             void *down_weight_ptr, void *down_scale_ptr, void *down_zero_ptr,
                             size_t layer_id) {
    std::cout << "Loading mlp shared experts " << layer_id << " from " << gate_weight_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        auto gate_w = weight->w_layers[layer_id].share_expert->gate->w;
        auto gate_s = weight->w_layers[layer_id].share_expert->gate->s;
        auto gate_z = weight->w_layers[layer_id].share_expert->gate->z;
        auto up_w = weight->w_layers[layer_id].share_expert->up->w;
        auto up_s = weight->w_layers[layer_id].share_expert->up->s;
        auto up_z = weight->w_layers[layer_id].share_expert->up->z;
        auto down_w = weight->w_layers[layer_id].share_expert->down->w;
        auto down_s = weight->w_layers[layer_id].share_expert->down->s;
        auto down_z = weight->w_layers[layer_id].share_expert->down->z;
        load_dist_linear(gate_weight_ptr, gate_scale_ptr, gate_zero_ptr, gate_w, gate_s, gate_z, weights->device_weights.size(), dev, weight->load_stream);
        load_dist_linear(up_weight_ptr, up_scale_ptr, up_zero_ptr, up_w, up_s, up_z, weights->device_weights.size(), dev, weight->load_stream);
        load_dist_linear(down_weight_ptr, down_scale_ptr, down_zero_ptr, down_w, down_s, down_z, weights->device_weights.size(), dev, weight->load_stream);
    }
}

void load_mlp_experts(DeepSeekV3Weights *weights,
                      void *gate_weight_ptr, void *gate_scale_ptr, void *gate_zero_ptr,
                      void *up_weight_ptr, void *up_scale_ptr, void *up_zero_ptr,
                      void *down_weight_ptr, void *down_scale_ptr, void *down_zero_ptr,
                      size_t layer_id, size_t expert_id) {
    std::cout << "Loading mlp expert " << layer_id << " expert " << expert_id
              << " from " << gate_weight_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        auto gate_w = weight->w_layers[layer_id].experts[expert_id]->gate->w;
        auto gate_s = weight->w_layers[layer_id].experts[expert_id]->gate->s;
        auto gate_z = weight->w_layers[layer_id].experts[expert_id]->gate->z;
        auto up_w = weight->w_layers[layer_id].experts[expert_id]->up->w;
        auto up_s = weight->w_layers[layer_id].experts[expert_id]->up->s;
        auto up_z = weight->w_layers[layer_id].experts[expert_id]->up->z;
        auto down_w = weight->w_layers[layer_id].experts[expert_id]->down->w;
        auto down_s = weight->w_layers[layer_id].experts[expert_id]->down->s;
        auto down_z = weight->w_layers[layer_id].experts[expert_id]->down->z;
        load_dist_linear(gate_weight_ptr, gate_scale_ptr, gate_zero_ptr, gate_w, gate_s, gate_z, weights->device_weights.size(), dev, weight->load_stream);
        load_dist_linear(up_weight_ptr, up_scale_ptr, up_zero_ptr, up_w, up_s, up_z, weights->device_weights.size(), dev, weight->load_stream);
        load_dist_linear(down_weight_ptr, down_scale_ptr, down_zero_ptr, down_w, down_s, down_z, weights->device_weights.size(), dev, weight->load_stream);
    }
}

static DeepSeekV3WeightLoader weight_loader = {
    // Global
    .load_input_embd = load_input_embd,
    .load_output_norm = load_output_norm,
    .load_output_embd = load_output_embd,
    // Attention
    .load_attn_norm = load_attn_norm,
    .load_attn_q_a_proj = load_attn_q_a_proj,
    .load_attn_q_a_layernorm = load_attn_q_a_layernorm,
    .load_attn_q_b_proj = load_attn_q_b_proj,
    .load_attn_kv_a_proj_with_mqa = load_attn_kv_a_proj_with_mqa,
    .load_attn_kv_a_layernorm = load_attn_kv_a_layernorm,
    .load_attn_kv_b_proj = load_attn_kv_b_proj,
    .load_attn_o_proj = load_attn_o_proj,
    // MLP
    .load_mlp_norm = load_mlp_norm,
    .load_mlp_dense = load_mlp_dense,
    .load_mlp_gate_weight = load_mlp_gate_weight,
    .load_mlp_gate_bias = load_mlp_gate_bias,
    .load_mlp_shared_experts = load_mlp_shared_experts,
    .load_mlp_experts = load_mlp_experts,
};

__C DeepSeekV3Weights *
createDeepSeekV3Weights(const DeepSeekV3Meta *meta,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids) {
    auto weights = new DeepSeekV3Weights(meta, device, ndev, dev_ids);
    return weights;
};

__C DeepSeekV3WeightLoader *
createDeepSeekV3WeightLoader() {
    return &weight_loader;
}

#include "jiuge_awq.hpp"

#include <cmath>

inline std::shared_ptr<Tensor> getSinTable(size_t dctx, size_t dh, float theta) {
    auto half_dh = dh / 2;
    auto unit = dsize(INFINI_DTYPE_F16);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));

            ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, INFINI_DTYPE_F16, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(size_t dctx, size_t dh, float theta) {
    auto half_dh = dh / 2;
    auto unit = dsize(INFINI_DTYPE_F16);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));

            ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, INFINI_DTYPE_F16, shape);
    std::free(table);
    return tensor;
}

JiugeAWQWeights::JiugeAWQWeights(
    const JiugeAWQMeta *meta,
    infiniDevice_t device,
    const std::vector<int> &dev_ids) : infinicore::WeightsLoader(device, dev_ids) {
    auto ndev = dev_ids.size();
    _device_weights.resize(ndev);
    infiniDtype_t dt_logits = meta->dt_logits;
    infiniDtype_t dt_norm_w = meta->dt_norm_w;
    size_t nlayer = meta->nlayer;
    size_t d = meta->d;
    size_t nh = meta->nh / ndev;
    size_t nkvh = meta->nkvh / ndev;
    size_t dh = meta->dh;
    size_t di = meta->di / ndev;
    size_t dctx = meta->dctx;
    size_t dvoc = meta->dvoc;
    size_t nbit = meta->nbit;
    size_t quant_group_size = meta->quant_group_size;

    for (size_t i = 0; i < ndev; i++) {
        RUN_INFINI(infinirtSetDevice(device, dev_ids[i]));

        auto weight = std::make_shared<JiugeAWQDeviceWeight>();
        _device_weights[i] = weight;

        auto w_in_embd = Tensor::weight(nullptr, dt_logits, {dvoc, d});
        this->resigter("model.embed_tokens.weight", w_in_embd, i);
        weight->w_in_embd = w_in_embd;

        auto w_out_norm = Tensor::weight(nullptr, dt_norm_w, {d});
        this->resigter("model.norm.weight", w_out_norm, i);
        weight->w_out_norm = w_out_norm;

        auto w_out_embd = Tensor::weight(nullptr, dt_logits, {dvoc, d})->permute({1, 0});
        this->resigter("lm_head.weight", w_out_embd, i);
        weight->w_out_embd = w_out_embd;

        weight->sin_table = getSinTable(dctx, dh, meta->theta);
        weight->cos_table = getCosTable(dctx, dh, meta->theta);

        for (size_t layer = 0; layer < nlayer; layer++) {

#define RIGISTER_LAYER_WEIGHT(W_NAME, W_VAR, W_SHAPE, W_DTYPE) \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, W_SHAPE);    \
    this->resigter(W_NAME, W_VAR, i);                          \
    weight->W_VAR.push_back(W_VAR);

            RIGISTER_LAYER_WEIGHT("model.layers." + std::to_string(layer) + ".input_layernorm.weight", w_attn_norm, {d}, dt_norm_w);

#define REGISTER_LAYER_QUANT_WEIGHT(W_NAME, W_VAR, W_IN, W_OUT)                                           \
    auto W_VAR = std::make_shared<QuantInt4Weight>();                                                     \
    W_VAR->w = Tensor::weight(nullptr, INFINI_DTYPE_I32, {W_IN, (W_OUT)*nbit / 32});                      \
    this->resigter(W_NAME + ".qweight", W_VAR->w, i);                                                     \
    W_VAR->s = Tensor::weight(nullptr, INFINI_DTYPE_F16, {(W_IN) / quant_group_size, (W_OUT)});           \
    this->resigter(W_NAME + ".scales", W_VAR->s, i);                                                      \
    W_VAR->z = Tensor::weight(nullptr, INFINI_DTYPE_I32, {(W_IN) / quant_group_size, (W_OUT)*nbit / 32}); \
    this->resigter(W_NAME + ".qzeros", W_VAR->z, i);                                                      \
    weight->W_VAR.push_back(W_VAR);

            REGISTER_LAYER_QUANT_WEIGHT("model.layers." + std::to_string(layer) + ".self_attn.q_proj", w_attn_q, d, nh * dh);
            REGISTER_LAYER_QUANT_WEIGHT("model.layers." + std::to_string(layer) + ".self_attn.k_proj", w_attn_k, d, nkvh * dh);
            REGISTER_LAYER_QUANT_WEIGHT("model.layers." + std::to_string(layer) + ".self_attn.v_proj", w_attn_v, d, nkvh * dh);
            RIGISTER_LAYER_WEIGHT("model.layers." + std::to_string(layer) + ".self_attn.q_proj.bias", b_attn_q, {nh * dh}, INFINI_DTYPE_F16);
            RIGISTER_LAYER_WEIGHT("model.layers." + std::to_string(layer) + ".self_attn.k_proj.bias", b_attn_k, {nkvh * dh}, INFINI_DTYPE_F16);
            RIGISTER_LAYER_WEIGHT("model.layers." + std::to_string(layer) + ".self_attn.v_proj.bias", b_attn_v, {nkvh * dh}, INFINI_DTYPE_F16);
            REGISTER_LAYER_QUANT_WEIGHT("model.layers." + std::to_string(layer) + ".self_attn.o_proj", w_attn_out, nh * dh, d);

            RIGISTER_LAYER_WEIGHT("model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight", w_ffn_norm, {d}, dt_norm_w);
            REGISTER_LAYER_QUANT_WEIGHT("model.layers." + std::to_string(layer) + ".mlp.gate_proj", w_ffn_gate, d, di);
            REGISTER_LAYER_QUANT_WEIGHT("model.layers." + std::to_string(layer) + ".mlp.up_proj", w_ffn_up, d, di);
            REGISTER_LAYER_QUANT_WEIGHT("model.layers." + std::to_string(layer) + ".mlp.down_proj", w_ffn_down, di, d);
        }
    }

#undef RIGISTER_LAYER_WEIGHT
#undef REGISTER_LAYER_QUANT_WEIGHT
}

__C struct ModelWeights *
createJiugeAWQWeights(const JiugeAWQMeta *meta,
                      infiniDevice_t device,
                      int ndev,
                      const int *dev_ids) {
    JiugeAWQWeights *weights = new JiugeAWQWeights(meta, device, std::vector<int>(dev_ids, dev_ids + ndev));
    return (struct ModelWeights *)weights;
}

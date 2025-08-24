#ifndef JIUGE_WEIGHT_HPP
#define JIUGE_WEIGHT_HPP

#include "jiuge_impl.hpp"

#include <cmath>
inline std::shared_ptr<Tensor> getInEmbd(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
}

inline std::shared_ptr<Tensor> getOutNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getOutEmbd(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({meta->dvoc, meta->d});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({meta->d, meta->dvoc});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape);
    }
}

inline std::shared_ptr<Tensor> getAttnNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnQKV(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, (nh + 2 * nkvh) / ndev * dh});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getAttnQKVBias(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * dsize(w->dt_mat);
    auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh});
    return Tensor::weight((char *)(w->attn_qkv_b[layer]) + offset, w->dt_mat, shape);
}

inline std::shared_ptr<Tensor> getAttnQNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_q_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnKNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_k_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnO(JiugeMeta const *meta,
                                        JiugeWeights const *w, size_t layer,
                                        size_t idev, size_t ndev) {
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * d * (nh / ndev * dh) * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, nh / ndev * dh});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({nh / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getFFNNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getFFNGateUp(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * (2 * di / ndev) * d * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({2 * di / ndev, d});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                              w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, 2 * di / ndev});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                              w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getFFNDown(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * d * (di / ndev) * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, di / ndev});
        return Tensor::weight((char *)(w->ffn_down[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({di / ndev, d});
        return Tensor::weight((char *)(w->ffn_down[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getSinTable(JiugeMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
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

inline std::shared_ptr<Tensor> getCosTable(JiugeMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
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

#endif

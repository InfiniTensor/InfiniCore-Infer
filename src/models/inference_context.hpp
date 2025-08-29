#pragma once

#include "../cache_manager/opcache_manager.hpp"

#include <cassert>

struct InferenceContext {
    infiniopHandle_t op_handle;
    std::shared_ptr<MemoryPool> memory_pool;
    CacheManager *cache_manager;
    infinirtStream_t stream;
    std::shared_ptr<Storage> workspace_storage;
    size_t current_workspace_size = 0;

    InferenceContext(infiniopHandle_t op_handle, std::shared_ptr<MemoryPool> memory_pool, CacheManager *cache_manager, infinirtStream_t stream);

    void ensure_workspace(size_t required_size);

    void add(std::shared_ptr<Tensor> c,
             std::shared_ptr<Tensor> a,
             std::shared_ptr<Tensor> b);
    void rmsnorm(std::shared_ptr<Tensor> y,
                 std::shared_ptr<Tensor> x,
                 std::shared_ptr<Tensor> w,
                 float epsilon);
    void gemm(std::shared_ptr<Tensor> c,
              std::shared_ptr<Tensor> a,
              std::shared_ptr<Tensor> b,
              float alpha, float beta);
    void rearrange(std::shared_ptr<Tensor> dst,
                   std::shared_ptr<Tensor> src);
    void rope(std::shared_ptr<Tensor> q,
              std::shared_ptr<Tensor> k,
              std::shared_ptr<Tensor> pos,
              std::shared_ptr<Tensor> sin,
              std::shared_ptr<Tensor> cos);
    void rope_v2(std::shared_ptr<Tensor> q,
                 std::shared_ptr<Tensor> k,
                 std::shared_ptr<Tensor> pos,
                 std::shared_ptr<Tensor> sin,
                 std::shared_ptr<Tensor> cos);
    void causalSoftmax(std::shared_ptr<Tensor> y,
                       std::shared_ptr<Tensor> x);

    void topkrouter(std::shared_ptr<Tensor> values,  // F32
                    std::shared_ptr<Tensor> indices, // I32
                    std::shared_ptr<Tensor> x,
                    std::shared_ptr<Tensor> correction_bias, // F32
                    float routed_scaling_factor,
                    size_t topk);

    void swiglu(std::shared_ptr<Tensor> out,
                std::shared_ptr<Tensor> up,
                std::shared_ptr<Tensor> gate);
    void randomSample(std::shared_ptr<Tensor> out,
                      std::shared_ptr<Tensor> prob,
                      float random_val, float top_p, uint32_t top_k, float temperature);

    void linear(std::shared_ptr<Tensor> c,
                std::shared_ptr<Tensor> a,
                std::shared_ptr<Tensor> b,
                float alpha, float beta,
                std::shared_ptr<Tensor> residual,
                std::shared_ptr<Tensor> bias);
    void dequant(std::shared_ptr<Tensor> weight,
                 std::shared_ptr<Tensor> in_w,
                 std::shared_ptr<Tensor> in_s,
                 std::shared_ptr<Tensor> in_z);
};

namespace {
thread_local InferenceContext *tls_inference_context = nullptr;
}

inline InferenceContext &getInferenceContext() {
    assert(tls_inference_context != nullptr && "InferenceContext not set for this thread");
    return *tls_inference_context;
}

inline void setInferenceContext(InferenceContext *ctx) {
    tls_inference_context = ctx;
}

inline void add(std::shared_ptr<Tensor> c, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    getInferenceContext().add(c, a, b);
}

inline void rmsnorm(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> x,
                    std::shared_ptr<Tensor> w, float epsilon) {
    getInferenceContext().rmsnorm(y, x, w, epsilon);
}

inline void gemm(std::shared_ptr<Tensor> c, std::shared_ptr<Tensor> a,
                 std::shared_ptr<Tensor> b, float alpha, float beta) {
    getInferenceContext().gemm(c, a, b, alpha, beta);
}

inline void rearrange(std::shared_ptr<Tensor> dst, std::shared_ptr<Tensor> src) {
    getInferenceContext().rearrange(dst, src);
}

inline void rope(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k,
                 std::shared_ptr<Tensor> pos, std::shared_ptr<Tensor> sin,
                 std::shared_ptr<Tensor> cos) {
    getInferenceContext().rope(q, k, pos, sin, cos);
}

inline void rope_v2(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k,
                    std::shared_ptr<Tensor> pos, std::shared_ptr<Tensor> sin,
                    std::shared_ptr<Tensor> cos) {
    getInferenceContext().rope_v2(q, k, pos, sin, cos);
}

inline void causalSoftmax(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> x) {
    getInferenceContext().causalSoftmax(y, x);
}

inline void topkrouter(std::shared_ptr<Tensor> values,  // F32
                       std::shared_ptr<Tensor> indices, // I32
                       std::shared_ptr<Tensor> x,
                       std::shared_ptr<Tensor> correction_bias, // F32
                       float routed_scaling_factor,
                       size_t topk) {

    getInferenceContext().topkrouter(values,  // F32
                                     indices, // I32
                                     x,
                                     correction_bias, // F32
                                     routed_scaling_factor,
                                     topk);
}

inline void swiglu(std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> up,
                   std::shared_ptr<Tensor> gate) {
    getInferenceContext().swiglu(out, up, gate);
}

inline void randomSample(std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> prob,
                         float random_val, float top_p, uint32_t top_k, float temperature) {
    getInferenceContext().randomSample(out, prob, random_val, top_p, top_k, temperature);
}

inline void linear(std::shared_ptr<Tensor> c, std::shared_ptr<Tensor> a,
                   std::shared_ptr<Tensor> b, float alpha, float beta,
                   std::shared_ptr<Tensor> residual, std::shared_ptr<Tensor> bias) {
    getInferenceContext().linear(c, a, b, alpha, beta, residual, bias);
}

inline void dequant_linear(std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> x,
                           std::shared_ptr<Tensor> w_w, std::shared_ptr<Tensor> w_s, std::shared_ptr<Tensor> w_z,
                           float alpha, float beta, std::shared_ptr<Tensor> residual, std::shared_ptr<Tensor> bias) {
    auto w = Tensor::buffer(x->dtype(), {x->shape()[1], out->shape()[1]}, getInferenceContext().memory_pool);
    getInferenceContext().dequant(w, w_w, w_s, w_z);
    getInferenceContext().linear(out, x, w, alpha, beta, residual, bias);
}

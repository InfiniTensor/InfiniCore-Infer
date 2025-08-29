#ifndef DEEPSEEK_V3_WEIGHTS_H
#define DEEPSEEK_V3_WEIGHTS_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stddef.h>
#include <stdint.h>

struct DeepSeekV3Weights;

// Function pointer signatures
typedef void (*load_global_fn)(DeepSeekV3Weights *, void *cpu_ptr);
typedef void (*load_layer_fn)(DeepSeekV3Weights *, void *cpu_ptr, size_t layer_id);
typedef void (*load_layer_linear_fn)(DeepSeekV3Weights *, void *weight_ptr, void *scale_ptr, void *zero_ptr, size_t layer_id);
typedef void (*load_layer_mlp_fn)(
    DeepSeekV3Weights *,
    void *gate_weight_ptr, void *gate_scale_ptr, void *gate_zero_ptr,
    void *up_weight_ptr, void *up_scale_ptr, void *up_zero_ptr,
    void *down_weight_ptr, void *down_scale_ptr, void *down_zero_ptr,
    size_t layer_id);
typedef void (*load_layer_expert_mlp_fn)(
    DeepSeekV3Weights *,
    void *gate_weight_ptr, void *gate_scale_ptr, void *gate_zero_ptr,
    void *up_weight_ptr, void *up_scale_ptr, void *up_zero_ptr,
    void *down_weight_ptr, void *down_scale_ptr, void *down_zero_ptr,
    size_t layer_id, size_t expert_id);

// Struct containing all weight loading functions
typedef struct {
    // Global
    load_global_fn load_input_embd;
    load_global_fn load_output_norm;
    load_global_fn load_output_embd;

    // Attention
    load_layer_fn load_attn_norm;
    load_layer_linear_fn load_attn_q_a_proj;
    load_layer_fn load_attn_q_a_layernorm;
    load_layer_linear_fn load_attn_q_b_proj;
    load_layer_linear_fn load_attn_kv_a_proj_with_mqa;
    load_layer_fn load_attn_kv_a_layernorm;
    load_layer_linear_fn load_attn_kv_b_proj;
    load_layer_linear_fn load_attn_o_proj;

    // MLP
    load_layer_fn load_mlp_norm;
    // MLP dense part
    load_layer_mlp_fn load_mlp_dense;

    // MLP sparse gating
    load_layer_fn load_mlp_gate_weight;
    load_layer_fn load_mlp_gate_bias;

    // Shared experts
    load_layer_mlp_fn load_mlp_shared_experts;

    // Per-expert functions
    load_layer_expert_mlp_fn load_mlp_experts;

} DeepSeekV3WeightLoader;

struct DeepSeekV3Model;

typedef struct {
    infiniDtype_t dt_logits;
    infiniDtype_t dt_norm;
    infiniDtype_t dt_quant_weight;
    infiniDtype_t dt_quant_scale;
    infiniDtype_t dt_quant_zero;
    infiniDtype_t dt_gate_weight;
    infiniDtype_t dt_gate_bias;

    size_t n_sparse_layer;
    size_t n_dense_layer;
    size_t d;
    size_t nh;
    size_t nkvh;
    size_t d_rope;
    size_t d_nope;
    size_t r_q;
    size_t r_kv;
    size_t d_qk;
    size_t d_v;

    float routed_scale;
    size_t nexperts;
    size_t kexperts;
    size_t di;
    size_t di_moe;
    size_t dctx;
    size_t dvoc;

    float epsilon;
    float rope_theta;
    uint32_t end_token;

} DeepSeekV3Meta;

//////////////////// APIs ///////////////////////
/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct DeepSeekV3Model *
createDeepSeekV3Model(const DeepSeekV3Meta *,
                      const DeepSeekV3Weights *);

__C DeepSeekV3Weights *
createDeepSeekV3Weights(const DeepSeekV3Meta *meta,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids);

__C __export DeepSeekV3WeightLoader *
createDeepSeekV3WeightLoader();

/// @brief 销毁模型
__C __export void destroyDeepSeekV3Model(struct DeepSeekV3Model *);

__C __export struct DeepSeekV3Cache *
createDeepSeekV3Cache(const struct DeepSeekV3Model *);

__C __export void
dropDeepSeekV3Cache(const struct DeepSeekV3Model *,
                    struct DeepSeekV3Cache *);

/// @brief 批次推理一轮，并采样出新的 token
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
inferBatchDeepSeekV3(struct DeepSeekV3Model *,
                     const uint32_t *tokens, uint32_t ntok,
                     const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                     struct DeepSeekV3Cache **caches,
                     const float *temperature, const uint32_t *topk, const float *topp,
                     uint32_t *output);

/// @brief 批次推理一轮，输出 output embedding 后的 logits
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param logits 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
forwardBatchDeepSeekV3(struct DeepSeekV3Model *,
                       const uint32_t *tokens, uint32_t ntok,
                       const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                       struct DeepSeekV3Cache **caches,
                       void *logits);

#endif // DEEPSEEK_V3_WEIGHTS_H
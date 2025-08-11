#ifndef MODEL_TINYMIX_H
#define MODEL_TINYMIX_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>
#include <infinicore.h>

#include <stdint.h>

typedef infiniDtype_t DataType;

struct TinyMixModel;
struct KVCache;

typedef struct
{
    uint32_t nlayer, nh, nkvh, d, di, dvoc, dh;
    uint32_t nexpert, topk;
    DataType dt_logits, dt_mat, dt_norm;
    float epsilon, theta;
    uint32_t dctx;
} TinyMixMeta;

typedef struct
{
    size_t nlayer;
    infiniDtype_t dt_norm, dt_mat;
    int transpose_linear_weights;
    const void *input_embd;
    const void *output_norm;
    const void *output_embd;
    const void *const *attn_norm;
    const void *const *attn_qkv;
    const void *const *attn_qkv_b;
    const void *const *attn_o;
    const void *const *ffn_norm;
    const void *const *const *ffn_gate_up;
    const void *const *const *ffn_down;
    const void *const *ffn_gate;
} TinyMixWeights;

//////////////////// APIs ///////////////////////
__C __export struct TinyMixModel *
createTinyMixModel(const TinyMixMeta *,
                 const TinyMixWeights *,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids);

__C __export void
destroyTinyMixModel(struct TinyMixModel *);

__C __export struct KVCache *
createTinyMixKVCache(const struct TinyMixModel *);

__C __export struct KVCache *
duplicateTinyMixKVCache(const struct TinyMixModel *,
                 const struct KVCache *, uint32_t seq_len);

__C __export void
dropTinyMixKVCache(const struct TinyMixModel *,
            struct KVCache *);

__C __export void
inferBatchTinyMix(struct TinyMixModel *,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           const float *temperature, const uint32_t *topk, const float *topp,
           uint32_t *output);

#endif

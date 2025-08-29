#ifndef DEEPSEEK_V3_IMPL_H
#define DEEPSEEK_V3_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct QuantLinearWeight {
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> s;
    std::shared_ptr<Tensor> z;
};

struct MLAWeight {
    std::shared_ptr<Tensor> kv_a_norm, q_a_norm;
    std::shared_ptr<QuantLinearWeight> kv_a_proj, kv_b_proj, o_proj, q_a_proj, q_b_proj;
};

struct GateWeight {
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> b;
};

struct MLPWeight {
    std::shared_ptr<QuantLinearWeight> gate, up, down;
};

struct LayerWeight {
    std::shared_ptr<Tensor> mla_norm;
    std::shared_ptr<MLAWeight> mla;
    std::shared_ptr<Tensor> mlp_norm;
    std::shared_ptr<MLPWeight> dense_mlp;
    std::shared_ptr<GateWeight> route;
    std::shared_ptr<MLPWeight> share_expert;
    std::vector<std::shared_ptr<MLPWeight>> experts;
};

struct DeepSeekV3DeviceWeights {
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;
    std::vector<LayerWeight> w_layers;
    infiniDevice_t device;
    int dev_id;
    infinirtStream_t load_stream;
};

struct DeepSeekV3Weights {
    std::vector<std::shared_ptr<DeepSeekV3DeviceWeights>> device_weights;

    DeepSeekV3Weights(const DeepSeekV3Meta *meta,
                      infiniDevice_t device,
                      int ndev,
                      const int *dev_ids);
};

struct DeepSeekV3DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<DeepSeekV3DeviceWeights> weights;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    struct DeepSeekV3Cache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct DeepSeekV3Model {
    DeepSeekV3Meta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeepSeekV3DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    DeepSeekV3Model(const DeepSeekV3Meta *, const DeepSeekV3Weights *weights);
};

struct DeepSeekV3Cache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> kv_pass, k_rot;
};

#endif

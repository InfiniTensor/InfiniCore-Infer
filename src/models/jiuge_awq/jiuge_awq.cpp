#include "jiuge_awq.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"

#include <random>
#include <thread>
#include <vector>

void createDeviceResource(DeviceResource *rsrc, const JiugeAWQMeta *meta,
                          std::shared_ptr<JiugeAWQDeviceWeight> weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        weights,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(DeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void inferDeviceBatch(const JiugeAWQMeta *meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, void *last_logits) {
    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh / ndev;
    auto nh = meta->nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta.dctx;
    auto dh = meta->dh;
    auto d = meta->d;
    auto dt_logits = meta->dt_logits;
    auto di = meta->di / ndev;
    auto dvoc = meta->dvoc;
    auto stream = rsrc.stream;
    auto weight = rsrc.weights;
    bool has_qkv_bias = meta->has_qkv_bias;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto q_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto k_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);
    auto v_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);

    auto gate_buf = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);
    auto up_buf = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);

    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       weight->w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }
    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, weight->w_attn_norm[layer], meta->epsilon);
        // qkv_proj
        dequant_linear(q_buf, logits_out,
                       weight->w_attn_q[layer]->w, weight->w_attn_q[layer]->s, weight->w_attn_q[layer]->z,
                       1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_q[layer] : nullptr);
        dequant_linear(k_buf, logits_out,
                       weight->w_attn_k[layer]->w, weight->w_attn_k[layer]->s, weight->w_attn_k[layer]->z,
                       1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_k[layer] : nullptr);
        dequant_linear(v_buf, logits_out,
                       weight->w_attn_v[layer]->w, weight->w_attn_v[layer]->s, weight->w_attn_v[layer]->z,
                       1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_v[layer] : nullptr);
        // rope
        rope_v2(q_buf->view({ntok, nh, dh}), q_buf->view({ntok, nh, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
        rope_v2(k_buf->view({ntok, nkvh, dh}), k_buf->view({ntok, nkvh, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = q_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = k_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});
            auto v = v_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][layer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][layer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(1, 0, seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_buf->slice(1, 0, seq_len * total_len)->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }
        // o_proj
        dequant_linear(logits_in, o_buf, weight->w_attn_out[layer]->w, weight->w_attn_out[layer]->s, weight->w_attn_out[layer]->z,
                       1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        // 2. FFN
        rmsnorm(logits_out, logits_in, weight->w_ffn_norm[layer], meta->epsilon);
        dequant_linear(gate_buf, logits_out,
                       weight->w_ffn_gate[layer]->w, weight->w_ffn_gate[layer]->s, weight->w_ffn_gate[layer]->z,
                       1.0, 0.0, nullptr, nullptr);
        dequant_linear(up_buf, logits_out,
                       weight->w_ffn_up[layer]->w, weight->w_ffn_up[layer]->s, weight->w_ffn_up[layer]->z,
                       1.0, 0.0, nullptr, nullptr);
        swiglu(gate_buf, up_buf, gate_buf);
        dequant_linear(logits_in, gate_buf,
                       weight->w_ffn_down[layer]->w, weight->w_ffn_down[layer]->s, weight->w_ffn_down[layer]->z,
                       1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }
    // Sample and Output
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, weight->w_out_norm, meta->epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(), dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        weight->w_out_norm,
                        meta->epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            std::random_device _rd;
            std::mt19937 gen(_rd());
            token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, req, 1)->view_as({}, {}),
                             prob_buf->slice(0, req, 1)->view_as({dvoc}, {1}),
                             random_val, topp[req], topk[req], temperature[req]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = uint32_t(result_cpu[req]);
            }
        }
    }
}

__C void
inferBatchJiugeAWQ(struct JiugeAWQModel *model,
                   const uint32_t *tokens, uint32_t ntok,
                   const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                   struct KVCache **kv_caches,
                   const float *temperature, const uint32_t *topk, const float *topp,
                   uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
forwardBatchJiugeAWQ(struct JiugeAWQModel *model,
                     const uint32_t *tokens, uint32_t ntok,
                     const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                     struct KVCache **kv_caches,
                     void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

void launchDevice(const JiugeAWQMeta *meta, std::shared_ptr<JiugeAWQDeviceWeight> weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createDeviceResource(rsrc, meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                         req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

JiugeAWQModel::JiugeAWQModel(const JiugeAWQMeta *meta, const ModelWeights *weights_) {
    auto weights = (JiugeAWQWeights *)(weights_);
    device = weights->device();
    dev_ids = weights->dev_ids();
    int ndev = int(dev_ids.size());
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);

    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, meta, weights->device_weights()[i], &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct JiugeAWQModel *
createJiugeAWQModel(const JiugeAWQMeta *meta,
                    const ModelWeights *weights) {
    JiugeAWQModel *model = new JiugeAWQModel(meta, weights);
    return model;
}

__C void destroyJiugeAWQModel(struct JiugeAWQModel *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}

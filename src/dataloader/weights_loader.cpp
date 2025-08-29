#include "weights_loader.hpp"
#include "infinicore_infer/weights_loader.h"

#include "../utils.hpp"

#include <infinirt.h>

namespace infinicore {

WeightsLoader::WeightsLoader(infiniDevice_t dev, const std::vector<int> &dev_ids) : _device(dev), _dev_ids(dev_ids) {
    _streams.resize(_dev_ids.size());
    _weights.resize(_dev_ids.size());
    for (int rank = 0; rank < int(_dev_ids.size()); rank++) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        _weights[rank] = std::unordered_map<std::string, std::shared_ptr<Tensor>>();
        RUN_INFINI(infinirtStreamCreate(&_streams[rank]));
    }
}
void WeightsLoader::resigter(const std::string &name, std::shared_ptr<Tensor> tensor, int rank) {
    _weights[rank][name] = tensor;
}
void WeightsLoader::load_weight(const std::string &name, const void *host_data) {
    for (int rank = 0; rank < int(_dev_ids.size()); rank++) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        auto it = _weights[rank].find(name);
        if (it == _weights[rank].end()) {
            std::cerr << "Weight " << name << " not found in rank " << rank << std::endl;
            std::abort();
        }

        _weights[rank][name]->load(host_data, _streams[rank]);
    }
    for (int rank = int(_dev_ids.size() - 1); rank >= 0; rank--) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        RUN_INFINI(infinirtStreamSynchronize(_streams[rank]));
    }
}
void WeightsLoader::load_distributed_weight(const std::string &name, const void *host_data, const std::vector<int> &ranks) {
    for (size_t i = 0; i < ranks.size(); i++) {
        int rank = ranks[i];
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        auto it = _weights[rank].find(name);
        if (it == _weights[rank].end()) {
            std::cerr << "Weight " << name << " not found in rank " << rank << std::endl;
            std::abort();
        }
        _weights[rank][name]->load((char *)host_data + i * _weights[rank][name]->numel() * dsize(_weights[rank][name]->dtype()), _streams[rank]);
    }
    for (int rank = int(_dev_ids.size() - 1); rank >= 0; rank--) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        RUN_INFINI(infinirtStreamSynchronize(_streams[rank]));
    }
}
void WeightsLoader::load_rank_weight(const std::string &name, const void *host_data, int rank) {
    auto it = _weights[rank].find(name);
    if (it == _weights[rank].end()) {
        std::cerr << "Weight " << name << " not found in rank " << rank << std::endl;
        std::abort();
    }
    RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
    _weights[rank][name]->load(host_data);
}
void WeightsLoader::finalize() {
    int dev_id;
    RUN_INFINI(infinirtGetDevice(nullptr, &dev_id));
    for (int rank = 0; rank < int(_dev_ids.size()); rank++) {
        RUN_INFINI(infinirtSetDevice(_device, _dev_ids[rank]));
        RUN_INFINI(infinirtStreamSynchronize(_streams[rank]));
        RUN_INFINI(infinirtStreamDestroy(_streams[rank]));
    }
    RUN_INFINI(infinirtSetDevice(_device, dev_id));
}
std::shared_ptr<Tensor> WeightsLoader::get(const std::string &name, int rank) {
    return _weights[rank][name];
}

} // namespace infinicore

__C void
loadModelWeight(struct ModelWeights *weights_, const char *name, void *data) {
    std::string name_str(name);
    // std::cout << "Loading weight: " << name_str << std::endl;
    auto weights = reinterpret_cast<infinicore::WeightsLoader *>(weights_);
    weights->load_weight(name_str, data);
}

__C void
loadModelWeightDistributed(struct ModelWeights *weights_, const char *name, void *data, int *ranks, int nrank) {
    std::string name_str(name);
    // std::cout << "Loading dist weight: " << name_str << std::endl;
    auto weights = reinterpret_cast<infinicore::WeightsLoader *>(weights_);
    std::vector<int> rank_vec(ranks, ranks + nrank);
    weights->load_distributed_weight(name_str, data, rank_vec);
}

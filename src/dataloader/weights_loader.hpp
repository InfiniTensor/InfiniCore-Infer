#ifndef WEIGHTS_LOADER_HPP
#define WEIGHTS_LOADER_HPP

#include "../tensor.hpp"

#include <unordered_map>
#include <vector>

namespace infinicore {
class WeightsLoader {
protected:
    std::vector<std::unordered_map<std::string, std::shared_ptr<Tensor>>> _weights;
    infiniDevice_t _device;
    std::vector<int> _dev_ids;
    std::vector<infinirtStream_t> _streams;

public:
    WeightsLoader(infiniDevice_t, const std::vector<int> &dev_ids);
    void resigter(const std::string &name, std::shared_ptr<Tensor> tensor, int rank = 0);
    void load_weight(const std::string &name, const void *host_data);
    void load_distributed_weight(const std::string &name, const void *host_data, const std::vector<int> &ranks);
    void load_rank_weight(const std::string &name, const void *host_data, int rank);
    void finalize();
    std::shared_ptr<Tensor> get(const std::string &name, int rank = 0);
    const std::vector<int> &dev_ids() const { return _dev_ids; }
    infiniDevice_t device() const { return _device; }
};
} // namespace infinicore

#endif // WEIGHTS_LOADER_HPP

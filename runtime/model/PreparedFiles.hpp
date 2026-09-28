#pragma once

#include "model/PreparedWeights.hpp"
#include "model/WeightStore.hpp"

#include <filesystem>
#include <string_view>

namespace splash::model {

// The prepared files of one source: each reused, or written now and
// published. check runs throughout, admitConversion on a cache miss and
// sourceUnchanged around every write, so no file of a modified source is
// published or used.
class PreparedFiles final {
public:
  PreparedFiles(PreparationCheck check, PreparationCheck admitConversion, PreparationCheck sourceUnchanged)
      : guards_{std::move(check), std::move(admitConversion), std::move(sourceUnchanged)} {}
  [[nodiscard]] std::filesystem::path prepare(const PreparedWeight &weight, const WeightWriter &write) const {
    return store_.prepare(weight, write, guards_);
  }
  // The prepared file mapped as a weight file of magic, layer and type.
  [[nodiscard]] WeightFile open(metal::MetalBackend &backend, const PreparedWeight &weight,
                                const WeightWriter &write, std::string_view magic, uint32_t layer,
                                uint32_t type) const {
    return WeightFile(backend, prepare(weight, write), weight.component, magic, layer, type, weight.key);
  }

private:
  PreparationGuards guards_;
  PreparedWeights store_;
};

} // namespace splash::model

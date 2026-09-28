#pragma once

#include "WeightStore.hpp"
#include "VisionLoader.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

// The current Qwen targets share one vision-tower architecture. Its language
// projection width belongs to VisionLayout, so the loader is independent of a
// particular text model and validates the package-selected width.
struct QwenVisionWeights final {
  ops::VisionWeights tensors;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

// The packed vision/model.bin of directory.
[[nodiscard]] QwenVisionWeights
loadQwenVisionWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      ops::VisionLayout layout = {});
// The same layout, prepared from an upstream source.
[[nodiscard]] QwenVisionWeights
loadQwenVisionWeights(metal::MetalBackend &backend, const VisionLoader &source);

} // namespace splash::model

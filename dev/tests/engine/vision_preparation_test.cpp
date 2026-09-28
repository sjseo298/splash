// Prepares the tiny vision source run_vision_preparation.py writes (depth 2,
// width 8, 2x2 patches) and prints the prepared file and its cache key.
//
//   vision-preparation mlx|gguf DIRECTORY cold|warm [EXPECTED]
//
// warm requires a cache hit. EXPECTED is an independently serialized file the
// prepared bytes and the size estimate must equal.

#include "TestFiles.hpp"
#include "model/VisionLoader.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace splash;

int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 5)
      throw std::runtime_error("usage: vision-preparation mlx|gguf DIRECTORY "
                               "cold|warm [EXPECTED]");
    const std::string format = argv[1], mode = argv[3];
    if ((format != "mlx" && format != "gguf") ||
        (mode != "cold" && mode != "warm"))
      throw std::runtime_error("invalid vision-preparation arguments");
    const auto source = format == "mlx" ? model::VisionSource::Mlx
                                        : model::VisionSource::Gguf;
    ops::VisionLayout layout;
    layout.depth = 2;
    layout.hiddenSize = 8;
    layout.patchSize = 2;
    layout.patchDimension = 24;
    layout.intermediateSize = 10;
    layout.paddedIntermediateSize = 16;
    layout.mergedHiddenSize = 32;
    layout.outputHiddenSize = 8;
    layout.heads = 2;
    layout.headDimension = 4;
    layout.positionGridSide = 2;
    const auto forbidden = [] {
      throw std::runtime_error("unexpected warm conversion");
    };
    const model::VisionLoader preparation(argv[2], source, layout, {},
                                          mode == "warm" ? model::PreparationCheck(forbidden)
                                                         : model::PreparationCheck());
    const auto path = preparation.prepare();
    if (model::VisionLoader(argv[2], source, layout, {}, forbidden).prepare() !=
        path)
      throw std::runtime_error("warm cache miss");
    if (argc == 5) {
      const auto actual = test::readFile(path), expected = test::readFile(argv[4]);
      const auto differs = std::mismatch(actual.begin(), actual.end(),
                                         expected.begin(), expected.end());
      if (differs.first != actual.end() || differs.second != expected.end())
        throw std::runtime_error(
            "prepared bytes differ from the oracle from byte " +
            std::to_string(differs.first - actual.begin()) + " (" +
            std::to_string(actual.size()) + " prepared, " +
            std::to_string(expected.size()) + " expected)");
      if (model::preparedVisionBytes(layout) != expected.size())
        throw std::runtime_error("vision size estimate differs from the oracle");
    }
    std::cout << path.string() << '\n' << preparation.weight().key << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

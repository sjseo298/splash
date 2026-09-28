#include "AffineQ4Fixture.hpp"
#include "tuning/MoeTuning.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace splash::metal;
using namespace splash::ops;
using namespace splash::ops::tuning;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

uint64_t aligned(uint64_t bytes, uint64_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

void devicePolicyPlans() {
  constexpr MoeShape shape{2048, 256, 8, 512};
  // Include unknown cores, a decode crossover, measured parts and a part
  // whose prefill crossover is beyond the 2048-row production chunk.
  constexpr std::array devices{std::array{0U, 512U}, std::array{1U, 26U},
      std::array{10U, 260U}, std::array{16U, 416U}, std::array{20U, 520U},
      std::array{40U, 1040U}, std::array{80U, 2080U}};
  for (uint32_t family : {9U, 10U}) {
    for (const auto &[cores, threshold] : devices) {
      splash::DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = cores;
      ExecutionPlans production(device);
      const auto check = [&](const MoeWorkload &workload) {
        production.install({});
        const auto lookup = [&] {
          return workload.phase == MoePhase::Prefill
              ? production.moePrefill(shape, workload.rows)
              : production.moeDecode(shape, workload.rows / 8);
        };
        const auto candidates = production.moeCandidates(workload);
        const auto baseline = lookup();
        // Family 9 decode plans run the four-simdgroup 8-row tiles; prefill
        // and every other family keep the shipped eight.
        const auto simdgroups = workload.phase == MoePhase::Decode && family == 9
            ? MoeExpertSimdgroups::Four : MoeExpertSimdgroups::Eight;
        require(candidates.front().configuration() == baseline.configuration() &&
                    baseline.configuration().expertTile == (workload.phase == MoePhase::Prefill
                        ? MoeExpertTile::M32 : MoeExpertTile::M8) &&
                    baseline.configuration().m8Simdgroups == simdgroups,
                "MoE measurement/reporting baseline differs from production");
        for (const auto &candidate : candidates) {
          const auto route = moeRouteTile(workload.rows, candidate.configuration().routeWideRows);
          const bool wide = workload.rows >= threshold;
          require(candidate.configuration().routeWideRows == threshold &&
                      route.rows == (wide ? 32U : 8U) && route.experts == (wide ? 128U : 32U) &&
                      candidate.configuration().m8Simdgroups == simdgroups,
                  "MoE candidate departed from device router or expert-tile policy");
          OperatorChoices choices;
          choices.moe.push_back({workload, candidate.configuration()});
          production.install(choices);
          const auto selected = lookup();
          require(selected.configuration() == candidate.configuration() &&
                      selected.rows() == candidate.rows() &&
                      selected.splitExperts() == candidate.splitExperts() &&
                      selected.maximumTiles() == candidate.maximumTiles(),
                  "installed MoE candidate differs from its measured plan");
          require(production.moeCandidates(workload).front().configuration() == baseline.configuration(),
                  "installed choice changed the shipped tuning baseline");
          // Imported expert choices must retain this device's router and
          // expert-tile policy.
          choices.moe.front().configuration.routeWideRows = threshold + 1;
          choices.moe.front().configuration.m8Simdgroups =
              simdgroups == MoeExpertSimdgroups::Four ? MoeExpertSimdgroups::Eight
                                                      : MoeExpertSimdgroups::Four;
          production.install(choices);
          require(lookup().configuration() == candidate.configuration(),
                  "installed MoE choice overrode the device router or tile policy");
        }
      };
      for (uint32_t rows : {1U, 512U, 2048U, std::min(threshold - 1, 2048U),
                            std::min(threshold, 2048U), std::min(threshold + 1, 2048U)})
        check({shape, rows, MoePhase::Prefill});
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        check({shape, lanes * 8, MoePhase::Decode});
    }
  }
}

void fixtureBounds() {
  for (const MoeShape shape : {MoeShape{256, 4, 2, 256},
                               MoeShape{2048, 256, 8, 512},
                               MoeShape{1024, 32, 4, 2048}}) {
    // One input/residual/output and two distribution references, regardless
    // of how many real weight views a caller supplies; scratch fields take
    // the independent maximum over both candidates.
    const auto expectedBytes = [&](const std::array<MoePlan, 2> &candidates,
                                   uint32_t rows) {
      const uint64_t rowBytes = uint64_t{rows} * shape.hiddenSize * 2;
      uint64_t expected = 3 * aligned(rowBytes, 256) + aligned(2 * rowBytes, 256);
      for (const MoeScratchField &field : kMoeScratchFields)
        expected += aligned(std::max(candidates[0].workspace().*field.bytes,
                                     candidates[1].workspace().*field.bytes), 256);
      return aligned(expected, 16 * 1024);
    };
    for (uint32_t cores : {0U, 1U, 20U, 80U}) {
      splash::DeviceCapabilities device;
      device.gpuCoreCount = cores;
      const ExecutionPlans plans(device);
      for (uint32_t rows : {1U, 7U, 8U, 24U, 31U, 32U, 33U, 128U, 2048U}) {
        const MoeWorkload workload{shape, rows, MoePhase::Prefill};
        require(moeTuningFixtureBytes(workload) ==
                    expectedBytes(plans.moeCandidates(workload), rows),
                "MoE fixture did not admit independent field maxima on every device");
      }
    }
    // The decode candidates are the prefill ones reordered, except that the
    // split M32 prefill plan widens expertOutput when the intermediate width
    // exceeds the hidden width; the fixture must follow its own candidates.
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const uint64_t decode = moeTuningFixtureBytes({shape, lanes * 8, MoePhase::Decode});
      require(decode == expectedBytes(ExecutionPlans({}).moeCandidates(
                                         {shape, lanes * 8, MoePhase::Decode}), lanes * 8),
              "MoE decode fixture did not admit independent field maxima");
      require(shape.expertIntermediateSize > shape.hiddenSize
                  ? decode <= moeTuningFixtureBytes({shape, lanes * 8, MoePhase::Prefill})
                  : decode == moeTuningFixtureBytes({shape, lanes * 8, MoePhase::Prefill}),
              "MoE fixture bound depends on candidate ordering");
    }
  }
  for (uint32_t rows : {0U, 1U, 7U, 9U, 25U, 40U}) {
    bool rejected = false;
    try {
      (void)moeTuningFixtureBytes({{256, 4, 2, 256}, rows, MoePhase::Decode});
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "MoE tuning accepted invalid decode geometry");
  }
  for (uint32_t representatives = 1;
       representatives <= kMaximumMoeTuningRepresentatives; ++representatives)
    for (double seconds : {0.00001, 0.0005, 0.001, 0.005, 0.01}) {
      const uint32_t requested = measurementBatchRepetitions(seconds);
      const uint32_t rings = std::min(16U / representatives,
          (requested + representatives - 1) / representatives);
      const uint32_t repetitions = rings * representatives;
      require(repetitions >= representatives && repetitions <= 16 &&
                  repetitions % representatives == 0,
              "MoE batching dropped a representative or exceeded its bound");
    }
}

MetalBuffer zeroed(MetalBackend &backend, uint64_t bytes) {
  auto buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, "moe-tuning-test");
  std::memset(buffer.contents(), 0, static_cast<size_t>(bytes));
  return buffer;
}

Q8Projection router(MetalBackend &backend, bool shared, uint32_t representative = 0) {
  constexpr uint32_t hidden = 256;
  constexpr uint64_t elements = 256 * hidden;
  Q8Projection result{{zeroed(backend, elements), zeroed(backend, elements / 32),
                        zeroed(backend, elements / 32)},
                       256, hidden};
  if (!shared) {
    auto *weights = static_cast<uint8_t *>(result.planes.weights.contents());
    auto *scales = static_cast<uint16_t *>(result.planes.scales.contents());
    for (uint32_t expert = 0; expert < 4; ++expert) {
      weights[expert * 64 + expert + representative * 4] = 1;
      scales[expert] = floatToBf16(4.0F);
    }
  }
  return result;
}

ExpertProjection experts(MetalBackend &backend, uint32_t count, uint32_t salt = 0) {
  constexpr uint32_t width = 256;
  constexpr uint64_t parameters = uint64_t{width} * width / 64;
  return splash::test::expertSlabs(
      backend, count, width, width, "moe-tuning-test", [&](uint32_t expert, const splash::test::AffineQ4Planes &planes) {
        uint32_t seed = 12345U + expert + salt;
        for (uint64_t offset = 0; offset < parameters * 32; ++offset) {
          seed = seed * 1664525U + 1013904223U;
          planes.weights[offset] = static_cast<uint8_t>(seed >> 24);
        }
        for (uint64_t index = 0; index < parameters; ++index) {
          planes.scales[index] = floatToBf16(0.0078125F);
          planes.biases[index] = floatToBf16(-0.05859375F);
        }
      });
}

void nativeMeasurement(const char *library) {
  MetalBackend backend(library);
  const MoeWorkload workload{{256, 4, 2, 256}, 24, MoePhase::Decode};
  MoeTuningInput input{workload, {MoeWeights{}}};
  uint32_t admissions = 0;
  const AllocationAdmission denied = [&](uint64_t bytes, const auto &) {
    ++admissions;
    require(bytes == moeTuningFixtureBytes(workload), "wrong admitted fixture bytes");
    return false;
  };
  const auto deniedResult = tuneMoe(backend, denied, input);
  require(admissions == 1 && !deniedResult.complete && !deniedResult.failure &&
              deniedResult.choice.configuration == ExecutionPlans(backend.capabilities())
                  .moeDecode(workload.shape, workload.rows / 8).configuration(),
          "allocation denial did not retain baseline");
  admissions = 0;
  const auto cancelled = tuneMoe(backend, denied, input, {}, {}, [] { return true; });
  require(!admissions && !cancelled.complete && !cancelled.failure &&
              cancelled.measurements.front().status == MeasurementStatus::Cancelled,
          "cancelled tuning allocated or submitted work");
  const auto pressured = tuneMoe(backend, denied, input, {}, [] { return true; });
  require(!admissions && !pressured.complete && !pressured.failure &&
              pressured.measurements.front().status == MeasurementStatus::UnderPressure,
          "pressured tuning allocated or submitted work");
  MeasurementOptions invalid;
  invalid.samplePairs = 1;
  const auto invalidResult = tuneMoe(backend, denied, input, invalid);
  require(!admissions && !invalidResult.complete &&
              invalidResult.measurements.front().status == MeasurementStatus::InvalidInput,
          "invalid measurement options reached allocation");
  for (size_t count : {size_t{0}, kMaximumMoeTuningRepresentatives + 1}) {
    const MoeTuningInput invalidCount{workload, std::vector<MoeWeights>(count)};
    const auto rejected = tuneMoe(backend, denied, invalidCount);
    require(!admissions && !rejected.complete && rejected.failure &&
                rejected.measurements.front().status == MeasurementStatus::RunFailed,
            "invalid representative count reached admission");
  }

  const AllocationAdmission allowed = [&](uint64_t bytes, const auto &allocate) {
    require(bytes == moeTuningFixtureBytes(workload), "wrong fixture reservation");
    allocate();
    return true;
  };
  const uint64_t emptyBytes = backend.memoryStats().allocatedBytes;
  const auto invalidWeights = tuneMoe(backend, allowed, input);
  require(!invalidWeights.complete && invalidWeights.failure &&
              invalidWeights.measurements.back().status == MeasurementStatus::RunFailed &&
              backend.memoryStats().allocatedBytes == emptyBytes,
          "invalid weight failure was hidden or leaked fixture backing");

  input.weights.clear();
  for (uint32_t representative = 0; representative < 3; ++representative) {
    const uint32_t salt = representative * 101;
    input.weights.push_back(AffineMoeWeights{router(backend, false, representative),
        experts(backend, 4, salt), experts(backend, 4, salt + 1),
        experts(backend, 4, salt + 2), experts(backend, 1, salt + 3),
        experts(backend, 1, salt + 4), experts(backend, 1, salt + 5),
        router(backend, true, representative)});
  }
  const uint64_t modelBytes = backend.memoryStats().allocatedBytes;
  MeasurementOptions options;
  options.maximumWallSeconds = 30;
  const auto result = tuneMoe(backend, allowed, input, options);
  require(result.complete && !result.failure && result.measurements.size() == 2,
          "complete native MoE sweep failed");
  require(result.representativeCount == 3 && result.repetitions >= 3 &&
              result.repetitions <= 16 && result.repetitions % 3 == 0,
          "MoE did not measure complete equal-weight representative rings");
  std::cout << "MoE measurement representatives=" << result.representativeCount
            << " repetitions=" << result.repetitions << '\n';
  for (uint32_t distribution = 0; distribution < 2; ++distribution) {
    const auto &measurement = result.measurements[distribution];
    require(measurement.candidate.value == 1 &&
                measurement.workload.value == distribution &&
                measurement.pairCount == options.samplePairs &&
                measurement.warmup.returnedCalls == options.warmupPairs * 2 &&
                measurement.measurement.returnedCalls == options.samplePairs * 2,
            "MoE lost paired timings or distribution identity");
    std::cout << "MoE measurement distribution=" << distribution
              << " pairs=" << measurement.pairCount
              << " baseline_gpu_ms="
              << measurement.gpuAssessment.baselineMedianSeconds * 1000
              << " candidate_gpu_ms="
              << measurement.gpuAssessment.candidateMedianSeconds * 1000
              << " baseline_wall_ms="
              << measurement.wallAssessment.baselineMedianSeconds * 1000
              << " candidate_wall_ms="
              << measurement.wallAssessment.candidateMedianSeconds * 1000
              << '\n';
  }
  std::array<WorkloadMeasurements, 2> gpu, wall;
  constexpr std::array<WorkloadId, 2> required{{{0}, {1}}};
  for (uint32_t distribution = 0; distribution < 2; ++distribution) {
    const auto &measurement = result.measurements[distribution];
    gpu[distribution] = {required[distribution], measurement.rawGpuSamples()};
    wall[distribution] = {required[distribution], measurement.rawWallSamples()};
  }
  const std::array gpuCandidates{CandidateMeasurements{{1}, gpu}};
  const std::array wallCandidates{CandidateMeasurements{{1}, wall}};
  const bool qualified =
      selectCandidate(gpuCandidates, required, options.policy).verdict ==
          SelectionVerdict::Selected &&
      selectCandidate(wallCandidates, required, options.policy).verdict ==
          SelectionVerdict::Selected;
  require(result.choice.configuration.expertTile ==
              (qualified ? MoeExpertTile::M32 : MoeExpertTile::M8),
          "MoE selected without both distributions and both timing metrics");
  require(backend.memoryStats().allocatedBytes == modelBytes,
          "MoE tuning retained temporary backing");
  // A malformed LAST representative must stop at the finite-value gate
  // before timing. Checking only the first layer would incorrectly pass.
  auto *slab = static_cast<uint8_t *>(input.weights.back().affine().sharedDown.packed.contents());
  auto *scale = reinterpret_cast<uint16_t *>(slab + 256 * 256 / 2);
  scale[0] = 0x7fc0;
  const auto notFinite = tuneMoe(backend, allowed, input, options);
  require(!notFinite.complete && notFinite.failure &&
              notFinite.measurements.size() == 1 &&
              notFinite.measurements.front().pairCount == 0 &&
              backend.memoryStats().allocatedBytes == modelBytes,
          "non-finite output entered measurement or leaked memory");
}

} // namespace

int main(int argc, char **argv) {
  try {
    fixtureBounds();
    devicePolicyPlans();
    if (argc != 2)
      throw std::runtime_error("usage: moe-tuning --cpu|metallib");
    if (std::string_view(argv[1]) != "--cpu")
      nativeMeasurement(argv[1]);
    std::cout << "MoE tuning tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MoE tuning test failed: " << error.what() << '\n';
    return 1;
  }
}

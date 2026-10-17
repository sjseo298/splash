#include "engine/MemoryGovernor.hpp"

#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

namespace {

// What compressing the anonymous pages frees: the share the compressor does
// not keep at its ratio of held to own pages, at most half of them.
uint64_t compressionSavings(const HostMemoryPages &pages) noexcept {
  if (!pages.compressor || pages.compressed / 2 >= pages.compressor) return pages.anonymous / 2;
  if (pages.compressed <= pages.compressor) return 0;
  return pages.anonymous -
         static_cast<uint64_t>(static_cast<unsigned __int128>(pages.anonymous) * pages.compressor / pages.compressed);
}

} // namespace

HostMemory estimateHostMemory(const HostMemoryPages &statistics, uint64_t pageSize,
                              bool compression) noexcept {
  // free_count includes the speculative pages, which external_page_count
  // also counts; wired file pages are in neither.
  if (!pageSize || statistics.speculative > statistics.free) return {};
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  uint64_t pages = statistics.free - statistics.speculative;
  for (uint64_t reclaimable : {statistics.fileBacked, statistics.purgeable}) {
    if (reclaimable > maximum - pages) return {};
    pages += reclaimable;
  }
  const uint64_t credit = compression ? compressionSavings(statistics) : 0;
  if (credit > maximum - pages || pages + credit > maximum / pageSize) return {};
  return {(pages + credit) * pageSize, (pages + credit / 2) * pageSize};
}

std::optional<HostMemory> queryHostMemory() noexcept {
  mach_port_t host = mach_host_self();
  vm_size_t pageSize = 0;
  vm_statistics64_data_t statistics{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t pageResult = host_page_size(host, &pageSize);
  kern_return_t statisticsResult = pageResult == KERN_SUCCESS
      ? host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&statistics),
                          &count)
      : pageResult;
  mach_port_deallocate(mach_task_self(), host);
  if (statisticsResult != KERN_SUCCESS || !pageSize) return std::nullopt;

  return estimateHostMemory(
      {.free = statistics.free_count,
       .speculative = statistics.speculative_count,
       .fileBacked = statistics.external_page_count,
       .purgeable = statistics.purgeable_count,
       .anonymous = statistics.internal_page_count,
       .compressor = statistics.compressor_page_count,
       .compressed = statistics.total_uncompressed_pages_in_compressor},
      pageSize, querySystemMemoryPressure().value_or(MemoryPressure::Critical) != MemoryPressure::Critical);
}

bool ignoreHostPressure() noexcept {
  const char *val = std::getenv("SPLASH_IGNORE_HOST_PRESSURE");
  return val && (std::strcmp(val, "1") == 0 ||
                 std::strcmp(val, "true") == 0 ||
                 std::strcmp(val, "yes") == 0 ||
                 std::strcmp(val, "on") == 0);
}

std::optional<MemoryPressure> querySystemMemoryPressure() noexcept {
  uint32_t level = 0;
  size_t size = sizeof(level);
  if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size,
                   nullptr, 0) != 0 || size != sizeof(level))
    return std::nullopt;
  switch (level) {
  case DISPATCH_MEMORYPRESSURE_NORMAL: return MemoryPressure::Normal;
  case DISPATCH_MEMORYPRESSURE_WARN: return MemoryPressure::Warning;
  case DISPATCH_MEMORYPRESSURE_CRITICAL: return MemoryPressure::Critical;
  default: return std::nullopt;
  }
}

MemoryGovernor::Reservation::Reservation(MemoryGovernor *owner, uint64_t bytes)
    : owner_(owner), bytes_(bytes) {}

MemoryGovernor::Reservation::~Reservation() { release(); }

MemoryGovernor::Reservation::Reservation(Reservation &&other) noexcept
    : owner_(other.owner_), bytes_(other.bytes_) {
  other.owner_ = nullptr;
  other.bytes_ = 0;
}

void MemoryGovernor::Reservation::release() noexcept {
  if (owner_ && bytes_)
    owner_->release(bytes_);
  owner_ = nullptr;
  bytes_ = 0;
}

MemoryGovernor::MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                               uint64_t hostReserveBytes, uint64_t hostSpareReserveBytes,
                               HostMemoryProvider hostMemory, uint64_t untrackedReserveBytes)
    : backend_(backend), limitBytes_(limitBytes), hostReserveBytes_(hostReserveBytes),
      hostSpareReserveBytes_(hostSpareReserveBytes), hostMemory_(std::move(hostMemory)),
      untrackedReserveBytes_(untrackedReserveBytes) {
  if (!limitBytes_) {
    throw std::invalid_argument("memory governor limit must be positive");
  }
  if (!hostReserveBytes_) {
    throw std::invalid_argument("host memory reserve must be positive");
  }
  if (!hostMemory_) {
    throw std::invalid_argument("host memory provider must be present");
  }
  if (chargedBytes(true) > limitBytes_) {
    throw metal::MetalAllocationError(
        "existing Metal allocations exceed memory governor limit",
        metal::AllocationFailure::EngineBudget);
  }
  // The bypass guarantees the model's context: startup proceeds even when
  // other applications hold most of the host's memory.
  if (!ignoreHostPressure()) {
    const std::optional<HostMemory> host = hostMemory_();
    if (!host || host->available <= hostReserveBytes_) {
      throw metal::MetalAllocationError(
          "host available memory does not satisfy the system reserve",
          metal::AllocationFailure::HostPressure);
    }
  }
}

uint64_t
MemoryGovernor::chargedBytes(bool refreshDevice) const noexcept {
  metal::MetalMemoryStats memory = refreshDevice
      ? backend_.refreshMemoryStats()
      : backend_.memoryStats();
  // The backend's buffers are charged in full, the rest of the device's
  // footprint only where it exceeds the untracked reserve.
  const uint64_t limit = std::numeric_limits<uint64_t>::max();
  const uint64_t accounted = untrackedReserveBytes_ <= limit - memory.allocatedBytes
      ? memory.allocatedBytes + untrackedReserveBytes_
      : limit;
  return std::max(accounted, memory.deviceCurrentAllocatedBytes);
}

std::optional<HostMemory> MemoryGovernor::sampleHost() const noexcept {
  try {
    return hostMemory_();
  } catch (...) {
    return std::nullopt;
  }
}

uint64_t MemoryGovernor::hostHeadroomBytes(const std::optional<HostMemory> &host,
                                           uint64_t reservedBytes) const noexcept {
  if (!host || host->spare <= hostSpareReserveBytes_)
    return 0;
  const uint64_t spareAfterReserve = host->spare - hostSpareReserveBytes_;
  return reservedBytes < spareAfterReserve ? spareAfterReserve - reservedBytes : 0;
}

std::optional<MemoryGovernor::Reservation>
MemoryGovernor::tryReserve(uint64_t bytes, metal::AllocationFailure &failure) {
  if (!bytes) {
    throw std::invalid_argument("memory reservation must be positive");
  }
  uint64_t observed = chargedBytes(true);
  bool overflows =
      reservedBytes_ > std::numeric_limits<uint64_t>::max() - bytes;
  uint64_t requested =
      overflows ? std::numeric_limits<uint64_t>::max() : reservedBytes_ + bytes;
  const std::optional<HostMemory> host = sampleHost();
  host_ = hostState(host);
  const MemoryPressure pressure = effectivePressure(host_);
  bool engineFits = !overflows && observed <= limitBytes_ &&
                    requested <= limitBytes_ - observed;
  // Growth leaves the warning margin free above the host's spare reserve,
  // stops while macOS warns and, while a request waits for memory, waits for
  // the recovery margin once the host has run short, unless a request in
  // service needs it (setServing) or host pressure is bypassed. The bypass
  // waives the headroom checks and every hold but a system-critical pressure.
  const uint64_t hostRoom = hostHeadroomBytes(host, 0);
  const bool hostRoomFits =
      requested <= hostRoom && hostRoom - requested >= kHostWarningMarginBytes;
  // A request that only the host headroom refuses waits for host memory
  // while the idle headroom may still clear the margin. Hold host pressure
  // so the paced reclaim frees toward the recovery margin for it.
  if (!ignoreHostPressure() && engineFits && !serving_ && !hostRoomFits)
    host_.constrained = true;
  const bool hostRefuses = pressure == MemoryPressure::Critical ||
      (!ignoreHostPressure() && !serving_ &&
       (systemPressure_ == MemoryPressure::Warning || !hostRoomFits ||
        growthHeld(host_)));
  if (hostRefuses || !engineFits) {
    // The host's refusal lifts with its pressure, the limit's only once
    // memory is freed: a refusal they share is the host's.
    failure = hostRefuses ? metal::AllocationFailure::HostPressure
                          : metal::AllocationFailure::EngineBudget;
    if (deniedReservations_ != std::numeric_limits<uint64_t>::max()) {
      ++deniedReservations_;
    }
    return std::nullopt;
  }
  reservedBytes_ = requested;
  return Reservation(this, bytes);
}

metal::AllocationAdmission MemoryGovernor::allocationAdmission() noexcept {
  return [this](uint64_t bytes, const std::function<void()> &allocate)
             -> metal::AllocationResult {
    metal::AllocationFailure failure;
    auto reservation = tryReserve(bytes, failure);
    if (!reservation)
      return failure;
    try {
      allocate();
    } catch (const metal::MetalAllocationError &error) {
      // Host headroom is an estimate; the driver can still deny the allocation.
      return error.failure();
    }
    return {};
  };
}

bool MemoryGovernor::setServing(bool serving) noexcept {
  return std::exchange(serving_, serving);
}

void MemoryGovernor::setRequestWaiting(bool waiting) noexcept {
  requestWaiting_ = waiting;
}

void MemoryGovernor::setPressure(MemoryPressure pressure) noexcept {
  systemPressure_ = pressure;
}

void MemoryGovernor::reclaimed(ReclaimOutcome outcome) noexcept {
  if (outcome == ReclaimOutcome::Untargeted)
    return;
  host_.reclaimExhausted = host_.constrained && outcome == ReclaimOutcome::Exhausted;
}

MemoryGovernorSnapshot MemoryGovernor::snapshot() const noexcept {
  const std::optional<HostMemory> host = sampleHost();
  return snapshot(host, hostState(host));
}

MemoryGovernorSnapshot MemoryGovernor::evaluate() noexcept {
  const std::optional<HostMemory> host = sampleHost();
  host_ = hostState(host);
  return snapshot(host, host_);
}

MemoryGovernorSnapshot
MemoryGovernor::snapshot(const std::optional<HostMemory> &hostMemory,
                         const HostState &host) const noexcept {
  uint64_t observed = chargedBytes();
  uint64_t used = observed;
  if (reservedBytes_ <= std::numeric_limits<uint64_t>::max() - used) {
    used += reservedBytes_;
  } else {
    used = std::numeric_limits<uint64_t>::max();
  }
  uint64_t hostHeadroom = hostHeadroomBytes(hostMemory, reservedBytes_);
  const MemoryPressure pressure = effectivePressure(host);
  bool hostGrowthAllowed = ignoreHostPressure() ||
      (systemPressure_ == MemoryPressure::Normal && !growthHeld(host) &&
       hostHeadroom >= kHostWarningMarginBytes);
  return {
      limitBytes_,
      observed,
      used < limitBytes_ ? limitBytes_ - used : 0,
      pressure,
      deniedReservations_,
      hostMemory.has_value(),
      hostMemory ? hostMemory->available : 0,
      hostReserveBytes_,
      hostMemory ? hostMemory->spare : 0,
      hostSpareReserveBytes_,
      hostHeadroom,
      systemPressure_,
      hostGrowthAllowed,
  };
}

MemoryGovernor::HostState
MemoryGovernor::hostState(const std::optional<HostMemory> &hostMemory) const noexcept {
  // The bypass resets the hysteresis: headroom neither constrains nor holds.
  if (ignoreHostPressure()) return HostState{};
  const uint64_t hostHeadroom = hostHeadroomBytes(hostMemory, reservedBytes_);
  HostState next;
  next.constrained = !hostMemory ||
      hostHeadroom < kHostWarningMarginBytes ||
      (host_.constrained && hostHeadroom < kHostRecoveryMarginBytes);
  next.reclaimExhausted = host_.reclaimExhausted && next.constrained;
  return next;
}

MemoryPressure
MemoryGovernor::effectivePressure(const HostState &host) const noexcept {
  // Availability and the system's warning control growth and paced reclaim.
  // Only the system's critical signal warrants dropping every evictable cache
  // entry. The bypass ignores the headroom and warning signals alike, keeping
  // only macOS's own critical refusal so its jetsam keeps the process alive.
  if (ignoreHostPressure())
    return systemPressure_ == MemoryPressure::Critical ? MemoryPressure::Critical : MemoryPressure::Normal;
  if (systemPressure_ == MemoryPressure::Critical)
    return MemoryPressure::Critical;
  if (host.constrained || systemPressure_ == MemoryPressure::Warning)
    return MemoryPressure::Warning;
  return MemoryPressure::Normal;
}

void MemoryGovernor::release(uint64_t bytes) noexcept {
  if (bytes > reservedBytes_) {
    reservedBytes_ = 0;
    return;
  }
  reservedBytes_ -= bytes;
}

std::optional<MemoryReclaimDirective> MemoryPressurePolicy::update(
    const MemoryGovernorSnapshot &snapshot, double nowMilliseconds,
    bool requestWaiting) noexcept {
  // The engine's own budget can run tight while the host stays calm: below a
  // one-gate margin on a budget that holds two of them, reclaim idle cache so
  // an in-flight or arriving request still gets its full model context. Only
  // budgets large enough to hold the margin take this pass; smaller limits
  // keep the upstream policy whole.
  constexpr uint64_t kEngineTargetMargin = 1ULL << 30; // 1 GiB
  const bool engineLowHeadroom =
      snapshot.limitBytes >= 2 * kEngineTargetMargin &&
      snapshot.headroomBytes < kEngineTargetMargin;

  if (snapshot.pressure == MemoryPressure::Normal && !requestWaiting &&
      !engineLowHeadroom) {
    nextReclaimMilliseconds_ = 0.0;
    return std::nullopt;
  }
  if (snapshot.pressure == MemoryPressure::Critical)
    return MemoryReclaimDirective{.critical = true};
  if (nowMilliseconds < nextReclaimMilliseconds_)
    return continued_.value_or(MemoryReclaimDirective{});
  continued_.reset();

  // Missing telemetry pauses allocation, but is not evidence that live
  // cache must be discarded. Empty extents can still be returned; a tight
  // engine budget or a waiting request still reclaims.
  if (!snapshot.hostMeasurementValid &&
      snapshot.systemPressure == MemoryPressure::Normal &&
      !engineLowHeadroom && !requestWaiting)
    return MemoryReclaimDirective{};

  // A pass a request waits for frees toward the recovery margin, where growth
  // reopens (hostState); one that nothing waits for only restores the warning
  // margin that growth leaves free: freeing on to the recovery margin would
  // discard cache nothing needs. macOS's warning asks caches to give memory
  // back, whatever the headroom: while it lasts, every pass returns as much
  // as a pass may, and keeps the resume point even for a waiting request,
  // whose growth the warning refuses as well, so that what the pass frees
  // would not reach it. The bypass waives the host's passes; then only the
  // engine's own margin reclaims.
  const bool warning = snapshot.systemPressure == MemoryPressure::Warning;
  uint64_t desired = 0;
  if (engineLowHeadroom) {
    // A budget that holds two one-GiB margins reclaims toward the lower one,
    // even while the host stays calm.
    desired = kEngineTargetMargin - snapshot.headroomBytes;
  } else if (!ignoreHostPressure()) {
    const uint64_t margin =
        requestWaiting ? kHostRecoveryMarginBytes : kHostWarningMarginBytes;
    const uint64_t deficit =
        snapshot.hostHeadroomBytes < margin ? margin - snapshot.hostHeadroomBytes : 0;
    desired = warning ? kHostWarningMarginBytes : deficit;
  }
  // The host samples every 500 ms. Allow counters to settle after a pass
  // with a target, but keep responding if another application continues
  // consuming memory. A pass with none leaves nothing to settle: a request
  // that starts waiting gets its pass at once. Recovering the last stretch to
  // the watermark is worth far less than the resume point it would otherwise
  // discard, so a pass keeps that publication and takes the rest; a waiting
  // request outranks it unless macOS's warning already refuses its growth.
  if (desired)
    nextReclaimMilliseconds_ = nowMilliseconds + 1000.0;

  return MemoryReclaimDirective{
      .targetBytes = std::min(desired, kHostWarningMarginBytes),
      .keepResumePoint = !requestWaiting || warning};
}

void MemoryPressurePolicy::reclaimed(const MemoryReclaimDirective &directive,
                                     const MemoryReclaimResult &result) noexcept {
  continued_.reset();
  // Every critical pass evicts everything again by itself.
  if (result.outcome != ReclaimOutcome::Pending || directive.critical)
    return;
  continued_ = directive;
  continued_->targetBytes -= std::min(result.releasedBytes, directive.targetBytes);
}

} // namespace splash::engine

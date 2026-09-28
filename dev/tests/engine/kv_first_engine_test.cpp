#include "TestImmediateTicket.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

class Backing final : public KvBacking {
public:
  explicit Backing(uint32_t pages) : resident_(pages) {}
  uint32_t pageCount() const noexcept override { return resident_.size(); }
  uint64_t bytesPerPage() const noexcept override { return 4096; }
  bool isResident(uint32_t page) const override { return resident_.at(page); }
  splash::metal::AllocationResult ensureResident(uint32_t page) override {
    ++growthAttempts;
    if (growthBlocked || (growthAllowed && !growthAllowed()))
      return allocationFailure;
    uint32_t first = extentFirstPage(page);
    for (uint32_t i = first; i < first + extentPageCount(page); ++i) {
      resident_.at(i) = true;
    }
    return true;
  }
  bool releaseBackingForPage(uint32_t page) override {
    uint32_t first = extentFirstPage(page);
    for (uint32_t i = first; i < first + extentPageCount(page); ++i) {
      resident_.at(i) = false;
    }
    releasePending = deferRelease;
    return true;
  }
  bool releaseReady() const noexcept override {
    if (completeReleaseOnPoll)
      releasePending = false;
    return !releasePending;
  }
  uint32_t extentFirstPage(uint32_t page) const override {
    return page - page % 4;
  }
  uint32_t extentPageCount(uint32_t page) const override {
    return std::min<uint32_t>(4, resident_.size() - extentFirstPage(page));
  }
  bool deferRelease = false;
  mutable bool releasePending = false;
  bool completeReleaseOnPoll = false;
  bool growthBlocked = false;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::Capacity;
  uint32_t growthAttempts = 0;
  std::function<bool()> growthAllowed;

private:
  std::vector<bool> resident_;
};

class State final : public CompositeState {
public:
  uint64_t bytes() const noexcept override { return 64; }
};

class DiskState final : public CompositeState {
public:
  uint64_t bytes() const noexcept override { return 64; }
  uint64_t residentBytes() const noexcept override { return 0; }
};

struct OffloadControl {
  bool ready = false;
  bool released = false;
};

// A state write in flight; its disk copy is a DiskState.
class OffloadTicket final : public StateOffload {
public:
  explicit OffloadTicket(std::shared_ptr<OffloadControl> control)
      : control_(std::move(control)) {}
  bool ready() const noexcept override { return control_->ready; }
  bool finish() override { return true; }
  const std::shared_ptr<const CompositeState> &state() const noexcept override {
    return disk_;
  }
private:
  std::shared_ptr<OffloadControl> control_;
  std::shared_ptr<const CompositeState> disk_ = std::make_shared<DiskState>();
};

class OffloadState final : public CompositeState {
public:
  explicit OffloadState(std::shared_ptr<OffloadControl> control) : control_(std::move(control)) {}
  ~OffloadState() override { control_->released = true; }
  uint64_t bytes() const noexcept override { return 64; }
  bool canOffload() const noexcept override { return true; }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override {
    return std::make_unique<OffloadTicket>(control_);
  }
private:
  std::shared_ptr<OffloadControl> control_;
};

struct RestoreControl {
  bool ready = false;
  bool success = true;
  bool cancelled = false;
  // No cache slot for the restored state's RAM copy.
  bool promotionDenied = false;
};

class RestoreTicket final : public StateRestore {
public:
  std::shared_ptr<RestoreControl> control;
  std::function<void()> commit;
  bool ready() const noexcept override { return control->ready; }
  bool finish() override {
    if (!control->success || control->cancelled) return false;
    commit();
    return true;
  }
  void cancel() noexcept override { control->cancelled = true; }
  std::shared_ptr<const CompositeState> snapshot() override {
    if (control->promotionDenied)
      return nullptr;
    return std::make_shared<State>();
  }
};

struct MaskOverlapState final {
  uint64_t requestId = 0;
  bool finishes = true;
  bool emitted = false;
  bool provided = false;
  bool abandoned = false;
};

class MaskOverlapTicket final : public ModelBatchTicket {
public:
  explicit MaskOverlapTicket(std::shared_ptr<MaskOverlapState> state)
      : state_(std::move(state)) {}

  std::vector<ModelMaskRequest> takeMaskRequests() override {
    if (state_->emitted)
      return {};
    state_->emitted = true;
    return {{state_->requestId, {10, 11, 12, 13, 14, 15, 16, 17}}};
  }
  bool ownsMaskWait(uint64_t id) const noexcept override {
    return state_->emitted && id == state_->requestId;
  }
  void abandonMask(uint64_t id) noexcept override {
    if (id == state_->requestId)
      state_->abandoned = true;
  }
  bool ready() const noexcept override {
    return state_->provided || state_->abandoned;
  }
  std::vector<ModelStepResult> wait() override {
    if (!ready())
      throw std::logic_error("mask-overlap ticket completed without input");
    return {{state_->requestId, 0, {42}, state_->finishes,
             DecodeStage::Regular, 7, 0}};
  }
  double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::shared_ptr<MaskOverlapState> state_;
};

// A decode command that stays in flight until the test releases it.
class HeldTicket final : public ModelBatchTicket {
public:
  HeldTicket(std::vector<ModelStepResult> results,
             std::shared_ptr<bool> released)
      : results_(std::move(results)), released_(std::move(released)) {}
  bool ready() const noexcept override { return *released_; }
  std::vector<ModelStepResult> wait() override { return std::move(results_); }
  double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<bool> released_;
};

class Executor final : public model::Model {
public:
  explicit Executor(
      uint32_t maximumCells = model::ExecutionLimits::maximumBatchWidth)
      : maximumCells(maximumCells) {}

  StateAdmission begin(const ModelRequest &request) override {
    ++beginAttempts;
    if (beginObserver) beginObserver();
    if (beginGrowthBlocked && beginGrowthBlocked())
      return {{}, StateFailure::MemoryPressure, beginAllocationFailure};
    if (deniedBegins) {
      --deniedBegins;
      return {{}, StateFailure::MemoryPressure};
    }
    for (uint32_t slot = 0; slot < maximumCells; ++slot) {
      const bool used = std::any_of(
          requests.begin(), requests.end(), [slot](const auto &entry) {
            return entry.second.resident && entry.second.slot == slot;
          });
      if (!used) {
        requests.emplace(request.id, Request{slot, 0, true});
        return {slot, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void suspend(uint64_t id) override {
    Request &entry = requests.at(id);
    if (!entry.resident)
      throw std::logic_error("request is already suspended");
    entry.resident = false;
    entry.position = 0;
    ++suspensions;
    if (physicalGrowthBlocked && unblockGrowthOnSuspend)
      *physicalGrowthBlocked = false;
  }
  StateAdmission resume(const ModelRequest &request) override {
    ++resumeAttempts;
    if (resumeDenied)
      return {{}, StateFailure::MemoryPressure};
    const uint64_t id = request.id;
    Request &entry = requests.at(id);
    for (uint32_t slot = 0; slot < maximumCells; ++slot) {
      const bool used = std::any_of(
          requests.begin(), requests.end(), [&](const auto &candidate) {
            return candidate.first != id && candidate.second.resident &&
                   candidate.second.slot == slot;
          });
      if (!used) {
        entry.slot = slot;
        entry.resident = true;
        entry.position = 0;
        entry.replaying = true;
        resumedPrompts.emplace_back(request.prompt.begin(), request.prompt.end());
        ++resumptions;
        return {slot, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void restore(uint64_t id, uint32_t length,
               std::shared_ptr<const CompositeState> state,
               bool restoreDraftState) override {
    if (!state)
      throw std::runtime_error("empty restore state");
    if (restoreObserver)
      restoreObserver();
    requests.at(id).position = length;
    restored += length;
    restoredDraft = restoreDraftState;
  }
  std::unique_ptr<StateRestore> beginRestore(
      uint64_t id, uint32_t length, std::shared_ptr<const CompositeState> state,
      bool restoreDraft, std::function<void()>) override {
    if (state->residentBytes()) {
      restore(id, length, std::move(state), restoreDraft);
      return {};
    }
    ++diskReads;
    auto ticket = std::make_unique<RestoreTicket>();
    ticket->control = restoreControl;
    ticket->commit = [this, id, length, state, restoreDraft] {
      restore(id, length, state, restoreDraft);
    };
    return ticket;
  }
  void setDraftContextPlan(uint64_t id, DraftContextPlan plan) override {
    plans[id] = std::move(plan);
  }
  std::vector<ModelStepResult> prefill(const BatchPlan &,
                                       std::span<const ModelBatchItem> items) {
    prefillWidths.push_back(static_cast<uint32_t>(items.size()));
    std::vector<ModelStepResult> result;
    for (const auto &item : items) {
      requests.at(item.requestId).position += item.tokenCount;
      prefillRows += item.tokenCount;
      ModelStepResult step{item.requestId, item.tokenCount, {}, false,
                           requests.at(item.requestId).replaying
                               ? replayDecodeStage : DecodeStage::Regular,
                           0, 0};
      if (prefillAnchor && !requests.at(item.requestId).replaying) {
        // Prefill selected a stop token or the last budgeted token: emitted
        // now, without a KV row, and the request never decodes.
        step.outputTokens = {42};
        step.outputTokensWithoutKv = 1;
        step.finished = *prefillAnchor;
      }
      result.push_back(std::move(step));
    }
    return result;
  }
  std::vector<ModelStepResult> decode(const BatchPlan &plan,
                                      std::span<const ModelBatchItem> items) {
    std::vector<ModelStepResult> result;
    for (const auto &item : items) {
      // The production runtime stores eight verify rows from the lane's
      // position; the engine must have covered them with page-table entries.
      if (uint64_t{item.pageTable.size()} * KvCache::pageTokens <
          item.logicalPosition + model::ExecutionLimits::targetVerifyRows) {
        throw std::invalid_argument("page_table_too_short");
      }
      if (plan.cohort == BatchCohort::Constrained &&
          plan.decodeStage == DecodeStage::RequestInitialMask) {
        result.push_back({item.requestId,
                          0,
                          {},
                          false,
                          DecodeStage::ApplyInitialMask,
                          0,
                          0});
      } else {
        const std::vector<uint32_t> tokens =
            item.requestId == poisonRequest && poisonToken
                ? std::vector<uint32_t>{*poisonToken}
                : std::vector<uint32_t>{42};
        ModelStepResult step{item.requestId, 0, tokens, decodeFinishes,
                             DecodeStage::Regular, 0, 0};
        step.outputTokensWithoutKv = decodeTokensWithoutKv;
        result.push_back(std::move(step));
      }
    }
    return result;
  }
  std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
         std::function<void()> completion) override {
    // Like every production batch command, this one carries the tier's
    // queued copies; the test finishes the transfers themselves.
    if (tier && tier->copiesQueued()) {
      tier->queued = false;
      ++carryingCommands;
    }
    // Every constrained cycle after the initial mask request waits for its
    // mask inside the ticket, as the production constrained ticket does.
    if (plan.kind == WorkKind::Decode &&
        plan.cohort == BatchCohort::Constrained &&
        plan.decodeStage != DecodeStage::RequestInitialMask) {
      overlap = std::make_shared<MaskOverlapState>();
      overlap->requestId = items.front().requestId;
      overlap->finishes = decodeFinishes;
      return std::make_unique<MaskOverlapTicket>(overlap);
    }
    if (plan.kind == WorkKind::Decode && holdDecodeUntil) {
      return std::make_unique<HeldTicket>(decode(plan, items), holdDecodeUntil);
    }
    if (plan.kind == WorkKind::Prefill && holdPrefillUntil) {
      return std::make_unique<HeldTicket>(prefill(plan, items), holdPrefillUntil);
    }
    return test::immediateTicket(plan.kind == WorkKind::Prefill
                                     ? prefill(plan, items)
                                     : decode(plan, items),
                                 completion);
  }
  // The production model copies the lane's state into a cache slot at its
  // current page-aligned boundary and returns nullptr when no slot is free
  // and the governor denies a new one. The fake denies the next
  // `deniedSnapshots` calls, or every call made at `denySnapshotAtBoundary`.
  std::shared_ptr<const CompositeState> snapshot(uint64_t id) override {
    ++snapshotAttempts;
    if (snapshotObserver)
      snapshotObserver();
    if (deniedSnapshots) {
      --deniedSnapshots;
      return nullptr;
    }
    if (denySnapshotAtBoundary &&
        *denySnapshotAtBoundary == requests.at(id).position) {
      return nullptr;
    }
    ++snapshots;
    return std::make_shared<State>();
  }
  // Without a cache slot the production model writes the lane's state to
  // the disk tier; the fake has one when `stateTier` is set, with quota for
  // every state.
  bool canSnapshotToDisk() const noexcept override { return stateTier != nullptr; }
  std::unique_ptr<StateOffload> snapshotToDisk(uint64_t, std::function<void()>) override {
    ++diskSnapshots;
    return std::make_unique<OffloadTicket>(stateTier);
  }
  uint64_t reclaimIdleState() noexcept override {
    const uint64_t released = reclaimableIdleStateBytes;
    reclaimableIdleStateBytes = 0;
    reclaimedIdleStateBytes += released;
    if (released && physicalGrowthBlocked)
      *physicalGrowthBlocked = false;
    return released;
  }
  void provideMask(uint64_t id, std::span<const uint32_t>) override {
    if (overlap && overlap->emitted && overlap->requestId == id)
      overlap->provided = true;
  }
  void end(uint64_t id) override { requests.erase(id); }
  // A command that carries only the tier's queued copies; the test finishes
  // the transfers themselves.
  std::unique_ptr<ModelBatchTicket>
  submitTransfers(std::function<void()> completion) override {
    if (!tier || !tier->copiesQueued())
      return nullptr;
    tier->queued = false;
    ++transferCommands;
    return test::immediateTicket({}, completion);
  }

  test::TestKvTier *tier = nullptr;
  uint32_t transferCommands = 0;
  uint32_t carryingCommands = 0;

  struct Request {
    uint32_t slot = 0;
    uint32_t position = 0;
    bool resident = false;
    bool replaying = false;
  };
  std::unordered_map<uint64_t, Request> requests;
  std::unordered_map<uint64_t, DraftContextPlan> plans;
  std::shared_ptr<RestoreControl> restoreControl = std::make_shared<RestoreControl>();
  uint32_t diskReads = 0;
  uint32_t prefillRows = 0;
  uint32_t restored = 0;
  uint32_t snapshots = 0;
  uint32_t snapshotAttempts = 0;
  uint32_t diskSnapshots = 0;
  std::shared_ptr<OffloadControl> stateTier;
  uint32_t deniedSnapshots = 0;
  std::optional<uint32_t> denySnapshotAtBoundary;
  uint32_t beginAttempts = 0;
  uint32_t deniedBegins = 0;
  uint32_t suspensions = 0;
  uint32_t resumptions = 0;
  uint32_t resumeAttempts = 0;
  bool resumeDenied = false;
  uint32_t maximumCells = model::ExecutionLimits::maximumBatchWidth;
  std::function<void()> beginObserver;
  std::function<void()> restoreObserver;
  std::function<void()> snapshotObserver;
  std::function<bool()> beginGrowthBlocked;
  std::vector<std::vector<uint32_t>> resumedPrompts;
  std::vector<uint32_t> prefillWidths;
  bool restoredDraft = false;
  metal::AllocationFailure beginAllocationFailure = metal::AllocationFailure::None;
  uint64_t reclaimableIdleStateBytes = 0;
  uint64_t reclaimedIdleStateBytes = 0;
  bool *physicalGrowthBlocked = nullptr;
  bool unblockGrowthOnSuspend = true;
  bool decodeFinishes = true;
  DecodeStage replayDecodeStage = DecodeStage::Regular;
  uint32_t decodeTokensWithoutKv = 0;
  // When set, the decode step emits poisonToken (instead of 42) for
  // poisonRequest, exercising the engine's output validation.
  uint64_t poisonRequest = 0;
  std::optional<uint32_t> poisonToken;
  // Set when prefill itself ends the request: the value is `finished` (stop).
  std::optional<bool> prefillAnchor;
  std::shared_ptr<bool> holdDecodeUntil;
  std::shared_ptr<bool> holdPrefillUntil;
  std::shared_ptr<MaskOverlapState> overlap;
};

class Events final : public EngineEventSink {
public:
  void batchCompleted(WorkKind, uint32_t, uint32_t, uint32_t, uint32_t,
                      uint32_t, double) override {}
  void started(uint64_t requestId, EngineCacheStatus cache, uint32_t matched,
               uint32_t) override {
    startIds.push_back(requestId);
    starts.emplace_back(std::move(cache), matched);
  }
  void promptProgress(uint64_t id, uint32_t processed) override {
    progress[id].push_back(processed);
  }
  void tokens(uint64_t id, std::span<const uint32_t> values) override {
    emitted += values.size();
    auto &transcript = outputs[id];
    transcript.insert(transcript.end(), values.begin(), values.end());
  }
  void maskRequested(uint64_t requestId,
                     std::span<const uint32_t> simulation) override {
    maskRequests.emplace_back(
        requestId, std::vector<uint32_t>(simulation.begin(), simulation.end()));
  }
  void completed(uint64_t id, EngineFinishReason, uint32_t prompt,
                 uint32_t completion, std::span<const float>) override {
    ++completedCount;
    usage[id] = {prompt, completion};
  }
  void failed(uint64_t, std::string code, std::string message,
              bool retryable) override {
    ++failedCount;
    failures.push_back(std::move(code));
    failureDetails.emplace_back(std::move(message), retryable);
  }
  void capacityExhausted(uint64_t, uint32_t, uint32_t, uint64_t) override {
    ++capacityExhaustedCount;
  }

  std::unordered_map<uint64_t, std::vector<uint32_t>> progress;
  std::vector<std::pair<EngineCacheStatus, uint32_t>> starts;
  std::vector<uint64_t> startIds;
  std::unordered_map<uint64_t, std::vector<uint32_t>> outputs;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> usage;
  std::vector<std::string> failures;
  std::vector<std::pair<std::string, bool>> failureDetails;
  uint32_t emitted = 0;
  uint32_t completedCount = 0;
  uint32_t failedCount = 0;
  uint32_t capacityExhaustedCount = 0;
  std::vector<std::pair<uint64_t, std::vector<uint32_t>>> maskRequests;
};

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

EngineRequest request(uint64_t id, const std::vector<uint32_t> &prompt) {
  EngineRequest result;
  result.id = id;
  result.prompt = prompt;
  result.maxNewTokens = 1;
  result.deadlineMilliseconds = 10000;
  return result;
}

void runUntilIdle(engine::Engine &engine) {
  for (uint32_t step = 0; step < 32 && !engine.idle(); ++step) {
    static_cast<void>(engine.tick(step + 1));
  }
  require(engine.idle(), "engine did not reach idle");
}

void testConcurrentColdPrefixesComputeOnce() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id) {
    std::vector<uint32_t> prompt(193, 7);
    std::fill(prompt.begin() + 160, prompt.end(), id + 10);
    engine.submit(request(id, prompt));
  }
  static_cast<void>(engine.tick(0));
  require(model.requests.size() == 1,
          "shared cold prefix allocated redundant active state cells");
  runUntilIdle(engine);
  require(model.prefillRows == 160 + 4 * 33 && model.restored == 3 * 160,
          "concurrent cold requests recomputed their shared prefix");
  require(events.completedCount == 4 && events.failedCount == 0 &&
              events.emitted == 4,
          "shared prefill lost independent completions");
}

void testSharedPrefillRebuildsTheMissingJunctionOnce() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(6530, 7)));
  runUntilIdle(engine);
  std::vector<uint32_t> branch(6575, 7);
  std::fill(branch.begin() + 6517, branch.end(), 8);
  {
    auto hit = cache.lookup(branch);
    require(hit.kvBoundary == 6496 && hit.resumeBoundary() == 0,
            "fixture did not recreate a KV-only internal branch");
  }
  const uint32_t before = model.prefillRows;
  for (uint32_t id = 2; id <= 5; ++id) {
    std::fill(branch.begin() + 6517, branch.end(), id + 10);
    engine.submit(request(id, branch));
  }
  runUntilIdle(engine);
  require(model.prefillRows - before == 6496 + 4 * (6575 - 6496) &&
              model.restored == 3 * 6496 && events.completedCount == 5,
          "concurrent internal branches each rebuilt the missing GDN state");
}

void testSharedPrefillReleasesDifferentJunctionsIndependently() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  for (uint32_t id = 2; id <= 3; ++id) {
    std::vector<uint32_t> branch(193, 7);
    std::fill(branch.begin() + (id == 2 ? 96 : 160), branch.end(), id + 10);
    engine.submit(request(id, branch));
  }
  static_cast<void>(engine.tick(0));
  static_cast<void>(engine.tick(1));
  static_cast<void>(engine.tick(2));
  require(model.requests.size() == 2 && events.startIds.back() == 2,
          "a ready junction waited for the producer's longer shared prefix");
  runUntilIdle(engine);
  require(model.prefillRows == 193 + 97 + 33 && events.completedCount == 3,
          "different shared boundaries were not restored independently");
}

void testSharedPrefillEvictedPublicationFallsBack() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  engine.submit(request(2, std::vector<uint32_t>(193, 7)));
  static_cast<void>(engine.tick(0));
  static_cast<void>(engine.tick(1));
  require(cache.reclaimOneState(),
          "published shared prefix was pinned against pressure reclamation");
  runUntilIdle(engine);
  require(events.completedCount == 2 && model.prefillRows == 386 &&
              cache.snapshot().activeRequests == 0,
          "evicted shared publication stranded its waiter");
}

void testSharedPrefillProducerFailureReleasesWaiters() {
  for (bool cancelled : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(193, 7)));
    engine.submit(request(2, std::vector<uint32_t>(193, 7)));
    static_cast<void>(engine.tick(0));
    require(engine.snapshot().scheduler.waitingPrefix == 1 &&
                engine.resourceWaitSnapshot(0).memory == 0 &&
                model.requests.size() == 1,
            "shared prefill waiter was admitted before its prefix was ready");
    if (cancelled)
      engine.cancel(1);
    else
      engine.failRequest(1, "test_failure", "producer failed");
    runUntilIdle(engine);
    require(events.outputs[2] == std::vector<uint32_t>{42} &&
                cache.snapshot().activeRequests == 0 && model.requests.empty(),
            "failed prefix producer stranded a waiter or leaked resources");
  }
}

void testSharedPrefillWaiterCancellationAndDeadline() {
  for (bool cancelled : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(193, 7)));
    auto waiter = request(2, std::vector<uint32_t>(193, 7));
    waiter.deadlineMilliseconds = 1;
    engine.submit(std::move(waiter));
    static_cast<void>(engine.tick(0));
    if (cancelled)
      engine.cancel(2);
    runUntilIdle(engine);
    require(model.beginAttempts == 1 && events.outputs[2].empty() &&
                events.outputs[1] == std::vector<uint32_t>{42} &&
                engine.snapshot().scheduler.waitingPrefix == 0,
            "expired prefix waiter allocated a cell or interrupted its producer");
  }
}

void testSharedPrefillFailedPublicationFallsBack() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  model.deniedSnapshots = 100;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  runUntilIdle(engine);
  require(events.completedCount == 4 && events.failedCount == 0 &&
              model.prefillRows == 4 * 193 && cache.snapshot().activeRequests == 0,
          "a missing prefix snapshot stranded dependent requests");
}

void testSharedPrefillDoesNotBlockUnrelatedWork() {
  for (bool images : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    for (uint32_t id = 1; id <= 4; ++id) {
      auto value = request(id, std::vector<uint32_t>(193, images ? 7 : id));
      if (images) {
        value.images = {{0, 16, 8, 8, id, id}};
        value.imagePixels.assign(value.images.front().pixelBytes(), 1);
      }
      engine.submit(std::move(value));
    }
    static_cast<void>(engine.tick(0));
    require(model.requests.size() == 4 &&
                engine.snapshot().scheduler.waitingPrefix == 0,
            "unrelated token or image prefixes were serialized");
    runUntilIdle(engine);
    require(model.prefillRows == 4 * 193 && events.completedCount == 4,
            "unrelated work incorrectly reused a shared state");
  }
}

void testSharedPrefillHonorsPriorityAndLateArrival() {
  for (bool foreground : {false, true}) {
    Backing backing(512);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    auto producer = request(1, std::vector<uint32_t>(5001, 7));
    producer.priority = RequestPriority::Background;
    engine.submit(std::move(producer));
    static_cast<void>(engine.tick(0));
    auto waiter = request(2, std::vector<uint32_t>(5001, 7));
    waiter.priority = foreground ? RequestPriority::Foreground
                                : RequestPriority::Background;
    engine.submit(std::move(waiter));
    static_cast<void>(engine.tick(1));
    static_cast<void>(engine.tick(2));
    require(model.requests.size() == (foreground ? 2u : 1u),
            "prefix admission ignored priority or a late arrival");
    runUntilIdle(engine);
    require(events.completedCount == 2 && events.failedCount == 0,
            "late prefix waiter did not finish");
    if (!foreground)
      require(model.prefillRows == 5010 && model.restored == 4992,
              "late arrival missed the producer's planned replay point");
  }
}

void testLateSharedPrefillExtendsTheProducerPlan() {
  for (bool denyCheckpoint : {false, true}) {
    Backing backing(512);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    if (denyCheckpoint)
      model.denySnapshotAtBoundary = 4096;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(6601, 7)));
    static_cast<void>(engine.tick(0));
    for (uint32_t id = 2; id <= 4; ++id) {
      std::vector<uint32_t> branch(6601, 7);
      std::fill(branch.begin() + 6500, branch.end(), id + 10);
      engine.submit(request(id, branch));
    }
    runUntilIdle(engine);
    require(events.completedCount == 4 && events.failedCount == 0 &&
                model.prefillRows == 6601 + 3 * (6601 - 6496) &&
                model.restored == 3 * 6496,
            "late siblings recomputed the prefix after the producer's checkpoint");
  }
}

void testSharedPrefillCapacityFailureDoesNotDeadlock() {
  Backing backing(4);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  for (uint32_t step = 0; step < 64 && !engine.idle(); ++step)
    static_cast<void>(engine.tick(step * 1000));
  require(engine.idle() && cache.snapshot().activeRequests == 0 &&
              model.requests.empty() && events.emitted == 0,
          "capacity failure stranded a shared prefix producer or waiter");
  engine.submit(request(5, std::vector<uint32_t>(33, 9)));
  runUntilIdle(engine);
  require(events.outputs[5] == std::vector<uint32_t>{42},
          "capacity failure prevented subsequent service");
}

void testColdPublishesReplayStateAndLazyJunctionCanRebuildIt() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, resources, executor, events);
  require(engine.snapshot().maximumContextTokens == 102400,
          "snapshot lost the configured context limit");
  std::vector<uint32_t> prompt(65);
  for (uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = i + 1;

  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(executor.prefillRows == 65 && executor.snapshots == 1 &&
              engine.snapshot().replayStatePublications == 1,
          "cold request did not retain its latest Page32 replay state");
  require(events.starts.size() == 1 &&
              events.starts[0].first == EngineCacheStatus::Miss,
          "cold request reported a cache hit");

  require(resources.reclaimCache(1, false) != 0,
          "test could not remove the latest replay state");
  engine.submit(request(2, prompt));
  runUntilIdle(engine);
  require(executor.restored == 0 && executor.prefillRows == 130 &&
              executor.snapshots == 2 &&
              engine.snapshot().junctionMaterializations == 1,
          "second request did not lazily materialize its proven KV junction");
  require(events.starts.size() == 2 &&
              events.starts[1].first == EngineCacheStatus::Miss,
          "KV-only junction replay was reported as a state hit");

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  require(executor.restored == 64 && executor.prefillRows == 131,
          "cache hit did not replay exactly one real input token");
  require(events.starts.size() == 3 &&
              events.starts[2] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "state-backed hit accounting is wrong");
  require(events.completedCount == 3 && events.failedCount == 0 &&
              events.emitted == 3,
          "request lifecycle did not complete cleanly");
}

void testConcurrentDuplicateStateSkipsSnapshotCapture() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prompt(65);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;

  engine.submit(request(101, prompt));
  engine.submit(request(102, prompt));
  runUntilIdle(engine);

  // The second request restores the first publication without reserving a
  // redundant state cell or capturing the same state again.
  const auto snapshot = engine.snapshot();
  require(executor.snapshotAttempts == 1 && executor.snapshots == 1,
          "duplicate state publication performed a second snapshot capture");
  require(snapshot.resources.stateCache.entries == 1 &&
              snapshot.resources.stateCache.publications == 1 &&
              snapshot.cacheHits == 1 && executor.prefillRows == 66 &&
              snapshot.replayStatePublications == 1 &&
              snapshot.replayStatePublicationFailures == 0,
          "duplicate state publication was not reused and accounted");
}

void testImageSpansKeyPrefixIdentity() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  // Image runs render as one placeholder id, so two prompts with different
  // images are token-identical; only the span digests differ.
  std::vector<uint32_t> prompt(65, 248056);
  auto withImage = [&](uint64_t id, uint64_t digest) {
    EngineRequest result = request(id, prompt);
    result.images = {{8, 16, 8, 8, digest, digest ^ 0xabcdULL}};
    result.imagePixels.assign(result.images[0].pixelBytes(), 1);
    return result;
  };

  engine.submit(withImage(1, 0x1111));
  runUntilIdle(engine);
  require(events.starts.size() == 1 &&
              events.starts[0].first == EngineCacheStatus::Miss,
          "image producer unexpectedly hit the cache");
  engine.submit(withImage(2, 0x2222));
  runUntilIdle(engine);
  require(events.starts.size() == 2 &&
              events.starts[1] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::Miss, 0},
          "a different image falsely matched the cached prefix");
  engine.submit(withImage(3, 0x1111));
  runUntilIdle(engine);
  require(events.starts.size() == 3 &&
              events.starts[2] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "an identical image did not reuse the cached prefix");
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(events.starts.size() == 4 &&
              events.starts[3] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::Miss, 0},
          "a text-only prompt matched an image-keyed prefix");

  EngineRequest malformed = withImage(5, 0x1111);
  malformed.images[0].tokens = 15;
  bool rejected = false;
  try {
    engine.submit(std::move(malformed));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected,
          "image span with the wrong merged token count was admitted");
  require(events.completedCount == 4 && events.failedCount == 0,
          "image request lifecycle did not complete cleanly");
}

void testOneRequestPublishesJunctionAndLatestReplayState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(5, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimCache(1, false) != 0,
            "test could not leave a KV-only shared prefix");
  }

  prompt.resize(161, 777);
  engine.submit(request(6, prompt));
  runUntilIdle(engine);

  const DraftContextPlan &plan = executor.plans.at(6);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 96 &&
              plan.boundaries[1].boundary == 160 &&
              plan.boundaries[2].boundary == 161,
          "junction, latest replay state, and active end were not ordered");
  require(executor.snapshots == 3 && snapshot.junctionMaterializations == 1 &&
              snapshot.replayStatePublications == 2 &&
              snapshot.resources.stateCache.entries == 2,
          "one request did not retain both sparse composite states");
}

// A denied snapshot at the latest replay boundary recycles the least recently
// used cached state, which is an older unrelated state, not the junction state
// this same request published one command earlier.
void testLatestReplayDenialRecyclesOlderStateNotTheJunction() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(7, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimCache(1, false) != 0,
            "test could not remove the old composite state");
  }
  engine.submit(request(70, std::vector<uint32_t>(65, 7000)));
  runUntilIdle(engine);
  const auto older = resources.snapshot();
  require(older.stateCache.entries == 1 && older.stateCache.evictions == 1,
          "older unrelated state was not left in the cache");

  prompt.resize(161, 888);
  engine.submit(request(8, prompt));
  require(engine.tick(1) && engine.tick(2),
          "request did not publish its junction state");
  require(engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 2,
          "lazy junction was not materialized in the first command");
  executor.deniedSnapshots = 1;
  runUntilIdle(engine);

  const DraftContextPlan &plan = executor.plans.at(8);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 96 &&
              plan.boundaries[1].boundary == 160 &&
              plan.boundaries[2].boundary == 161,
          "boundaries were not armed without reservation");
  require(executor.deniedSnapshots == 0 &&
              snapshot.recycledStatePublications == 1 &&
              snapshot.replayStatePublications == 3 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.resources.stateCache.evictions == 2 &&
              snapshot.resources.stateCache.entries == 2,
          "latest-state denial did not recycle exactly one older state");

  // The junction state survived the recycle: the original prompt resumes
  // from it, while the recycled prompt is back to a KV-only junction.
  const uint32_t restored = executor.restored;
  prompt.resize(97);
  engine.submit(request(9, prompt));
  runUntilIdle(engine);
  require(executor.restored == restored + 96 &&
              events.starts.back() ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 96},
          "latest-state denial discarded the independent junction state");
  engine.submit(request(71, std::vector<uint32_t>(65, 7000)));
  runUntilIdle(engine);
  require(events.starts.back().first == EngineCacheStatus::Miss &&
              engine.snapshot().junctionMaterializations == 2,
          "recycled state was not the older unrelated one");
}

void testCancellationAfterJunctionDiscardsLaterState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(9, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimCache(1, false) != 0,
            "test could not remove the old composite state");
  }

  prompt.resize(161, 999);
  engine.submit(request(10, prompt));
  require(engine.tick(1) && engine.tick(2),
          "request did not publish its first sparse state");
  const auto published = engine.snapshot();
  require(published.resources.stateCache.entries == 1 &&
              published.junctionMaterializations == 1 &&
              executor.snapshotAttempts == 2,
          "junction publication did not land in the first command");
  engine.cancel(10);
  runUntilIdle(engine);
  // The armed 160 boundary is dropped with the lane: no snapshot, no
  // failure counted, and the junction state stays cached.
  const auto cancelled = engine.snapshot();
  require(executor.snapshotAttempts == 2 &&
              cancelled.resources.stateCache.entries == 1 &&
              cancelled.resources.stateCache.publications == 2 &&
              cancelled.replayStatePublications == 1 &&
              cancelled.replayStatePublicationFailures == 0 &&
              cancelled.cancelled == 1 && executor.requests.empty() &&
              cancelled.resources.activeRequests == 0,
          "cancellation leaked or removed the wrong sparse state");
}

// Nothing is reserved at admission, so a boundary the model cannot snapshot
// costs exactly one attempt: the failure is counted under its purpose, the
// request is served normally, and with an empty cache nothing is recycled.
void testDeniedSnapshotCostsOnlyThatAttempt() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.deniedSnapshots = 100;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prompt(65, 7);

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  const DraftContextPlan &plan = executor.plans.at(3);
  require(plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64 &&
              plan.boundaries[1].boundary == 65 &&
              plan.draftContextRows() == prompt.size(),
          "replay boundary was not armed without reservation");
  require(executor.snapshotAttempts == 1 && executor.snapshots == 0 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              engine.snapshot().junctionMaterializationFailures == 0 &&
              engine.snapshot().recycledStatePublications == 0 &&
              resources.snapshot().stateCache.evictions == 0 &&
              events.completedCount == 1,
          "latest replay-state denial was counted incorrectly");

  // The first pass left a usable KV chain but no state. The second lookup is
  // therefore a real lazy junction; its denied snapshot is counted as such
  // while the request replays through the KV prefix normally.
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(executor.snapshotAttempts == 2 && executor.snapshots == 0 &&
              engine.snapshot().junctionMaterializationFailures == 1 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              executor.prefillRows == 130 && events.completedCount == 2 &&
              events.failedCount == 0,
          "denied lazy junction failed the request or was not counted");

  // The denial is transient: the next lane over the same prefix lands it.
  executor.deniedSnapshots = 0;
  engine.submit(request(5, prompt));
  runUntilIdle(engine);
  require(executor.snapshots == 1 &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 1,
          "junction was not materialized once snapshots were possible");
}

// A snapshot the model denies once at a replay boundary lands by recycling
// the least recently used cached state: one eviction, the same number of
// entries, and the recycled slot now holds the new lane's state.
void testDeniedSnapshotRecyclesLruStateAndRetries() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> older(65, 1);
  const std::vector<uint32_t> newer(65, 2);
  engine.submit(request(1, older));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.stateCache.evictions == 0,
          "recycle fixture did not cache the older state");

  executor.deniedSnapshots = 1;
  engine.submit(request(2, newer));
  runUntilIdle(engine);
  const auto recycled = engine.snapshot();
  require(executor.snapshotAttempts == 3 && executor.snapshots == 2 &&
              recycled.recycledStatePublications == 1 &&
              recycled.replayStatePublications == 2 &&
              recycled.replayStatePublicationFailures == 0 &&
              events.completedCount == 2,
          "denied snapshot was not retried after recycling a state");
  require(recycled.resources.stateCache.evictions == 1 &&
              recycled.resources.stateCache.entries == 1 &&
              recycled.resources.stateCache.publications == 2 &&
              recycled.resources.stateCache.bytes == cached.stateCache.bytes,
          "recycling changed the cached state footprint");

  // The surviving state belongs to the new lane: re-sending its prompt is a
  // state hit, while the older prompt is back to a KV-only junction.
  engine.submit(request(3, newer));
  runUntilIdle(engine);
  require(executor.restored == 64 &&
              events.starts.back() ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "recycled slot does not hold the new lane's state");
  engine.submit(request(4, older));
  runUntilIdle(engine);
  require(executor.restored == 64 &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              engine.snapshot().junctionMaterializations == 1,
          "recycled state was not the least recently used one");
}

// When the retry after recycling is denied as well, the boundary fails and
// the engine has paid exactly one cached state for it. Persistent denial
// never drains the rest of the cache.
void testPersistentSnapshotDenialRecyclesAtMostOneState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  for (uint64_t id

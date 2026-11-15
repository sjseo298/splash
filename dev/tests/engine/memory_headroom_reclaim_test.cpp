#include "TestMetalMemory.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryLedger.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr uint64_t kMiB = 1024ULL * 1024;
constexpr uint64_t kGiB = 1024ULL * 1024 * 1024;

void require(bool condition, const char *msg) {
  if (!condition) {
    std::cerr << "FAIL: " << msg << std::endl;
    std::exit(1);
  }
}

class CacheVictimSource final : public VictimSource {
public:
  struct Entry {
    uint64_t key = 0;
    uint64_t bytes = 0;
    ReclaimClass cls = ReclaimClass::Ordinary;
  };

  std::vector<Entry> entries;
  uint64_t releasedBytes = 0;

  [[nodiscard]] std::string_view name() const noexcept override { return "cache"; }

  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &query, const Victim *after) const override {
    bool seenAfter = after == nullptr;
    for (const Entry &entry : entries) {
      if (!seenAfter) {
        if (entry.key == after->key)
          seenAfter = true;
        continue;
      }
      if (entry.cls == ReclaimClass::InUse && !query.rights.inUse)
        continue;
      return Victim{.key = entry.key, .bytes = entry.bytes, .cls = entry.cls, .kind = VictimKind::Memory};
    }
    return std::nullopt;
  }

  FreeResult take(const VictimQuery &, const Victim &victim) override {
    auto it = std::ranges::find(entries, victim.key, &Entry::key);
    if (it == entries.end())
      return {.outcome = FreeOutcome::Short};
    const uint64_t bytes = it->bytes;
    entries.erase(it);
    releasedBytes += bytes;
    if (test::metalStatistics().allocatedBytes >= bytes)
      test::metalStatistics().allocatedBytes -= bytes;
    if (test::metalStatistics().deviceCurrentAllocatedBytes >= bytes)
      test::metalStatistics().deviceCurrentAllocatedBytes -= bytes;
    return {.outcome = FreeOutcome::Met, .nowBytes = bytes};
  }
};

// 1. Test del Bypass de presión de memoria del Host
void testHostPressureBypass() {
  std::cout << "[TEST 1] Verificando bypass de presión de host..." << std::endl;

  // Por defecto, sin la variable de entorno, ignoreHostPressure() debe ser false
  ::unsetenv("SPLASH_IGNORE_HOST_PRESSURE");
  require(!ignoreHostPressure(), "ignoreHostPressure() debe ser false por defecto");

  // Al activar la variable, debe ser true
  ::setenv("SPLASH_IGNORE_HOST_PRESSURE", "1", 1);
  require(ignoreHostPressure(), "ignoreHostPressure() debe ser true con SPLASH_IGNORE_HOST_PRESSURE=1");

  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  // Even when host available memory is below hostReserveBytes, constructor proceeds with bypass.
  MemoryGovernor governor(
      backend, 8 * kGiB, 2 * kGiB, 2 * kGiB,
      [] { return std::optional<HostMemory>(HostMemory{512 * kMiB, 512 * kMiB}); }, 0);

  governor.setPressure(MemoryPressure::Warning);
  const MemoryGovernorSnapshot warnSnap = governor.evaluate();
  require(warnSnap.pressure == MemoryPressure::Normal,
          "Con bypass activo, Warning del sistema debe reportarse como Normal");
  require(warnSnap.hostGrowthAllowed && governor.admits(ClaimClass::Growth),
          "Con bypass activo, el crecimiento debe seguir permitido bajo Warning del host");

  const ClaimId growthClaim = governor.ledger().open({.cls = ClaimClass::Growth});
  {
    const MemoryGovernor::Scope scope(governor, growthClaim);
    const metal::AllocationResult res = governor.allocationAdmission()(64 * kMiB, [] {});
    require(static_cast<bool>(res),
            "Con bypass activo, una reserva de crecimiento bajo Warning debe ser admitida");
  }
  governor.ledger().close(growthClaim);

  governor.setPressure(MemoryPressure::Critical);
  const MemoryGovernorSnapshot critSnap = governor.evaluate();
  require(critSnap.pressure == MemoryPressure::Critical && !critSnap.hostGrowthAllowed,
          "Bajo presión Critical del sistema, incluso con bypass debe rechazarse crecimiento");

  ::unsetenv("SPLASH_IGNORE_HOST_PRESSURE");
  std::cout << "   ✓ ignoreHostPressure() conmuta correctamente y protege Critical del SO." << std::endl;
}

// 2. Test de política de desalojo bajo consumo alto de memoria Metal
void testEngineHeadroomReclaim() {
  std::cout << "[TEST 2] Verificando desalojo proactivo con margen bajo en Metal..." << std::endl;

  // Simular el escenario reportado con bypass activo:
  // Metal Limit: 28,271,624,192 bytes (~26.33 GiB)
  // Consumo actual: 28,246,425,600 bytes
  // Margen libre restante: ~25 MB (25,198,592 bytes)
  ::setenv("SPLASH_IGNORE_HOST_PRESSURE", "1", 1);
  const uint64_t limit = 28271624192ULL;
  const uint64_t resident = 28246425600ULL;
  const uint64_t headroom = limit - resident; // ~25 MB

  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = resident;
  test::metalStatistics().deviceCurrentAllocatedBytes = resident;

  metal::MetalBackend backend("unused");
  MemoryGovernor governor(
      backend, limit, 2 * kGiB, 2 * kGiB,
      [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); }, 0);

  CacheVictimSource cache;
  // Add 16 ordinary 64 MiB units (1 GiB) and 1 in-use unit that must NOT be touched.
  for (uint64_t i = 1; i <= 16; ++i)
    cache.entries.push_back({.key = i, .bytes = 64 * kMiB, .cls = ReclaimClass::Ordinary});
  cache.entries.push_back({.key = 99, .bytes = 64 * kMiB, .cls = ReclaimClass::InUse});
  governor.ledger().addSource(cache);

  // Escenario A: Margen de Metal bajo (< 1 GiB) con bypass activo
  static_cast<void>(governor.controlPass(1000.0));
  const uint64_t expectedMinTarget = kGiB - headroom;
  require(cache.releasedBytes >= expectedMinTarget,
          "Debe desalojar caché ordinaria hasta recuperar el margen objetivo de 1 GiB en Metal");
  require(std::ranges::any_of(cache.entries, [](const CacheVictimSource::Entry &e) { return e.key == 99; }),
          "No debe desalojar entradas en uso durante un pase proactivo de margen");

  std::cout << "   ✓ Desalojo activado: recuperados "
            << (cache.releasedBytes / kMiB) << " MiB para restaurar el margen de Metal." << std::endl;

  // Escenario B: Memoria con margen amplio (> 1 GiB)
  const uint64_t beforeHealthy = cache.releasedBytes;
  test::metalStatistics().allocatedBytes = 20ULL * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 20ULL * kGiB;
  cache.entries.push_back({.key = 100, .bytes = 64 * kMiB, .cls = ReclaimClass::Ordinary});
  static_cast<void>(governor.controlPass(3000.0));
  require(cache.releasedBytes == beforeHealthy,
          "Con margen de Metal holgado (> 1 GiB), no debe disparar desalojos innecesarios");

  ::unsetenv("SPLASH_IGNORE_HOST_PRESSURE");
  std::cout << "   ✓ En estado saludable (> 1 GiB libre), la caché permanece intacta sin desalojos espurios." << std::endl;
}

} // namespace

int main() {
  std::cout << "==========================================================" << std::endl;
  std::cout << "  TEST UNITARIO: Consumo y Desalojo de Memoria en Splash" << std::endl;
  std::cout << "==========================================================" << std::endl;

  testHostPressureBypass();
  testEngineHeadroomReclaim();

  std::cout << "\n>>> TODAS LAS PRUEBAS DE CONSUMO DE MEMORIA PASARON CON ÉXITO (100%) <<<" << std::endl;
  return 0;
}

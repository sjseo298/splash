#include "engine/MemoryGovernor.hpp"
#include <iostream>
#include <cassert>
#include <cstdlib>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool condition, const char *msg) {
  if (!condition) {
    std::cerr << "FAIL: " << msg << std::endl;
    std::exit(1);
  }
}

// 1. Test del Bypass de presión de memoria del Host
void testHostPressureBypass() {
  std::cout << "[TEST 1] Verificando bypass de presión de host..." << std::endl;

  // Por defecto, sin la variable de entorno, ignoreHostPressure() debe ser false
  ::unsetenv("SPLASH_IGNORE_HOST_PRESSURE");
  require(!ignoreHostPressure(), "ignoreHostPressure() debe ser false por defecto");

  // Al activar la variable, debe ser true
  ::setenv("SPLASH_IGNORE_HOST_PRESSURE", "1", 1);
  require(ignoreHostPressure(), "ignoreHostPressure() debe ser true con SPLASH_IGNORE_HOST_PRESSURE=1");

  std::cout << "   ✓ ignoreHostPressure() conmuta correctamente con la variable de entorno." << std::endl;
}

// 2. Test de política de desalojo bajo consumo alto de memoria Metal
void testEngineHeadroomReclaim() {
  std::cout << "[TEST 2] Verificando política de desalojo con margen bajo en Metal..." << std::endl;

  // Simular el escenario exacto reportado en los logs:
  // Metal Limit: 28.27 GB (28,271,624,192 bytes)
  // Consumo actual: 28.24 GB (28,246,425,600 bytes)
  // Margen libre restante: ~25 MB (25,198,592 bytes)
  const uint64_t limit = 28271624192ULL;
  const uint64_t resident = 28246425600ULL;
  const uint64_t headroom = limit - resident; // 25 MB

  MemoryGovernorSnapshot snap;
  snap.limitBytes = limit;
  snap.observedResidentBytes = resident;
  snap.headroomBytes = headroom;
  snap.pressure = MemoryPressure::Normal; // Presión host reportada normal por el bypass
  snap.systemPressure = MemoryPressure::Normal;
  snap.hostMeasurementValid = true;
  snap.growthAllowed = true;

  MemoryPressurePolicy policy;

  // Escenario A: Petición en curso pero no bloqueada, margen bajo (< 1 GiB)
  const double t0 = 1000.0;
  auto directiveA = policy.update(snap, t0, /*requestWaiting=*/false);

  require(directiveA.reclaimEmptyKvExtents,
          "Debe ordenar desalojo de extents KV vacíos cuando el margen en Metal es < 1 GiB");
  require(!directiveA.evictAllUnpinnedPrefixes,
          "No debe purgar indiscriminadamente toda la caché si no hay presión crítica");
  require(directiveA.targetBytes > 0,
          "El volumen objetivo de bytes a desalojar debe ser positivo");
  
  // El objetivo debe ser restaurar el margen objetivo de 1 GiB
  const uint64_t expectedTarget = (1ULL << 30) - headroom; // 1 GiB - 25 MB = ~999 MB
  require(directiveA.targetBytes == expectedTarget,
          "El objetivo a desalojar debe cubrir exactamente el déficit para recuperar 1 GiB");
  require(directiveA.keepResumePoint,
          "Debe intentar conservar el punto de reanudación si la petición no está esperando");

  std::cout << "   ✓ Desalojo activado: solicita recuperar " 
            << (directiveA.targetBytes / (1024 * 1024)) << " MB para restaurar el margen de Metal." << std::endl;

  // Escenario B: Petición atascada esperando recursos (requestWaiting = true)
  const double t1 = 2000.0;
  auto directiveB = policy.update(snap, t1, /*requestWaiting=*/true);

  require(directiveB.reclaimEmptyKvExtents,
          "Debe ordenar desalojo cuando hay una petición esperando");
  require(!directiveB.keepResumePoint,
          "Cuando hay una petición esperando (requestWaiting=true), debe ceder el punto de reanudación para priorizar la petición");

  std::cout << "   ✓ Priorización de solicitud: cede puntos de reanudación ociosos para dar paso al prompt." << std::endl;

  // Escenario C: Memoria con margen amplio (> 1 GiB) y sin peticiones esperando
  MemoryGovernorSnapshot healthySnap = snap;
  healthySnap.observedResidentBytes = 20ULL << 30; // 20 GiB usados de 28 GiB
  healthySnap.headroomBytes = healthySnap.limitBytes - healthySnap.observedResidentBytes; // 8 GiB libres

  auto directiveC = policy.update(healthySnap, 3000.0, /*requestWaiting=*/false);
  require(!directiveC.reclaimEmptyKvExtents,
          "Con margen de Metal holgado (> 1 GiB), no debe disparar desalojos innecesarios");
  require(directiveC.targetBytes == 0,
          "El objetivo a desalojar debe ser 0 cuando la memoria está holgada");

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

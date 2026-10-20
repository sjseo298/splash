# Architecture

Splash is a local inference engine for Apple silicon. It serves OpenAI- and
Anthropic-compatible HTTP APIs for coding agents and chat applications,
accelerated by DFlash 2 speculative decoding, purpose-built Metal kernels,
optional Neural Engine prefill, and automatic memory planning.

This document describes the structure of the codebase. For configuration,
the API reference, internals deep-dives, building, and testing, see
[DEVELOPMENT.md](DEVELOPMENT.md).

## Layers

```
                ┌────────────────────────────────────────────────────┐
                │  Clients: coding agents, OpenAI/Anthropic SDKs,    │
                │  browser chat page (server/chat.html)              │
                └──────────────────────┬─────────────────────────────┘
                                       │ HTTP (localhost, stdlib server)
┌──────────────────────────────────────▼─────────────────────────────┐
│  server/  (Python 3.12–3.14)                                       │
│  routing · protocol adaptation · request prep · streaming          │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ binary frame protocol over stdio
                       │ (magic "SPLH", version 10, little-endian)
┌──────────────────────▼─────────────────────────────────────────────┐
│  runtime/  (C++17 / Objective-C++, one binary: build/splash)       │
│                                                                    │
│  engine/   scheduler · memory admission · KV tiers · SSD cache     │
│  model/    weight loading · target/draft execution · vision        │
│  ops/      operators (attention, MoE, GDN, linears, sampling…)     │
│  metal/    Metal backend · kernel ABI · kernels/{prefill,decode,   │
│            shared,common} · precompiled metallib                   │
│  ane/      Neural Engine client + Metal→ANE handoff (prefill FFN)  │
└────────────────────────────────────────────────────────────────────┘
```

Everything below the HTTP layer is a single native process. Python is the
protocol and product layer; the native runtime owns execution.

## Entry points

- `./splash` — shell wrapper on `PATH`. Follows its own symlink to the
  checkout, selects Python 3.12–3.14 (preferring `.venv`), and execs
  `install/launcher.py`.
- `install/launcher.py` — `splash serve` (build the engine on first use,
  install the model, then run `python -m server.server` in the foreground)
  and `splash <agent>` (configure opencode, claude, codex, hermes, or pi to
  point at the local server).
- `server/server.py` — the running server; `server/connections.py` bounds
  HTTP ingress, `server/errors.py` owns each API's error shape.

## server/ (Python)

One package, relative imports only. Responsibilities by module:

| Module | Owns |
| --- | --- |
| `server.py` | HTTP routing, startup, signal handling, the chat page |
| `frontend.py` | request preparation: history, templates, images, PDFs, constraints |
| `backend.py` | native request lifecycles: submit, cancel, crash-loop recovery and relaunch with backoff, thinking-end selection |
| `runtime.py` | multiplexed subprocess client for the native protocol; calls under the pending limit go straight to the child; Python's executor handles CPU grammar masks |
| `protocol.py` | encodes client frames, decodes engine events (`runtime/engine/Protocol.*` does the reverse) |
| `output.py` | incremental parser for generated text; one parser serves streamed and complete responses |
| `constraints.py` | compiles token constraints (grammar masks for structured output and tool calls) |
| `judgments.py` | finite-choice prompts, validation, typed answer math |
| `api_shapes.py` | response shapes: OpenAI Chat/Completions/Responses, Anthropic Messages |
| `thinking.py`, `tokenization.py`, `images.py`, `documents.py`, `document_worker.py` | reasoning budgets, tokenizer access, image and PDF input |
| `latency.py`, `metrics.py`, `lru.py`, `origins.py`, `http_security.py`, `crash_trace.py`, `diagnostics.py`, `schema_validation.py`, `tool_schema.py`, `json_codec.py` | request latency tracking, `/metrics`, response cache, CORS and security, crash tracing, status display, JSON Schema validation, tool-call policy |
| `serve_options.py` | the options `splash serve` shares with the server; stdlib-only so the launcher can parse them before `.venv` exists |

The server binds localhost without authentication by default; LAN access,
auth, and browser-origin options are server configuration (DEVELOPMENT.md,
Server configuration).

## runtime/ (native)

Built into one binary, `build/splash`, plus the precompiled kernel library
`build/splash.metallib` — no Xcode or local kernel tuning at runtime.
`runtime/main.mm` boots the process: parses native arguments, sets up the
fd transport over stdio, keeps the Mac awake while requests are in flight
(IOPMLib), and observes memory pressure on a dispatch queue while all
allocation and reclaim decisions stay on the control loop.

Dependency direction is enforced by `make architecture-check`:
`metal/` is the bottom layer, `ops/` sits above it, `model/` above `ops/`,
and `engine/` at the top. The engine stays model-agnostic — only the
assembly sources (`Bootstrap`, `RuntimeResources`) may touch a concrete
model — and no production source contains client-specific behavior or
depends on test code.

### engine/ — scheduling and resource control

- `NativeRuntime` — the failure boundary: translates protocol frames and
  events at the `Engine` boundary; any exception reports one
  `EngineUnhealthy` event, closes the connection, and exits.
- `Protocol` — binary frames: 16-byte header (magic `SPLH`, u16 protocol
  version, u16 frame type, u64 payload length), little-endian payloads,
  frame types for request/cancel/status/release and start/tokens/done/error
  events. The byte layout is pinned by
  `dev/tests/engine/protocol_golden.txt`.
- `Engine` — request submission and cancellation, prefill checkpoints
  (every two draft windows), decode/prefill interleaving (`decodeShare`),
  resource waits with bounded backoff.
- `Scheduler`, `KvCache`, `KvPool`, `KvPageTier`, `KvTier` — paged KV cache
  (INT8 by default, BF16 optional) with page-level residency and extents.
- `MemoryGovernor`, `MemoryPlan`, `MemoryControl`, `MemoryAudit`,
  `ReleasableMemory`, `ThermalState` — automatic memory planning, host
  growth admission, idle release (default: release after 10 minutes),
  reclaim under pressure, thermal throttling.
- `Cache`, `CacheDirectory`, `WriteBehind`, `DiskSpace`, `DiskLabels`,
  `RecencyOrder`, `StateCache` — the SSD tier: write-behind of KV and
  recurrent states to a bounded disk cache (auto up to 32 GiB), persistent
  across restarts on request.
- `FdTransport`, `Bootstrap`, `NativeArguments`, `Status`, `Json`, `Types` —
  the stdio transport, startup wiring, status schema, small utilities.

### model/ — weights and execution

- Loading: every source format (safetensors, GGUF, MLX checkpoints) has a
  loader that validates metadata and plans the target images, and a writer
  that materializes them: `SafetensorsTargetLoader`, `GgufTargetLoader`,
  `DraftCheckpointLoader`, `VisionLoader`, all writing through
  `GgufPreparation` (GGUF repack without requantization, draft quantization,
  in-memory only — no on-disk conversion). `WeightImages` holds the
  finished images in `MDGG0001` layout in dedicated Metal buffers; sources
  are read uncached (`F_NOCACHE`) one thread per core. Golden hashes in
  `dev/tests/fixtures/weight-goldens/goldens.json` pin the emitted bytes.
- `ModelFactory` plans all loaders before any image is written, so every
  source's metadata is validated first; `ModelDescriptor` reads the
  installation's `model.json`.
- Execution: `Runtime` runs a prepared step; `Qwen3_8` and `Qwen3_6Moe`
  implement the supported targets, `DFlashDraft` the speculative draft,
  `QwenVision` the vision tower, `ExecutionPlans` the per-phase command
  plans; `QwenState`, `StateLayout`, `StateTransfer`, `GgufImage*`,
  `SafetensorsCheckpoint`, `WeightSource`, `WeightStore`, `Crc32c` support
  state and weight plumbing.

### ops/ — operators

One operator per concern, each exposing the workspace and shape contracts
that `model/` and `engine/` size through: `PagedAttention` / `DraftAttention`
(paged and draft attention), `Linear` / `LinearGguf` (dense and GGUF
quantized projections, SG-matrix layouts), `MoE`, `GDN` (gated delta-net
recurrent layers), `RoPE`, `Normalization`, `Embedding`, `Sampling`,
`Vision`, `RowCopy`, `AneFfn` (+ calibration and measurement),
`DraftSelector`, `ExecutionPlans`, `PageStorage`.

### metal/ — GPU backend

- `MetalBackend` builds `CommandGraph`s per phase with a
  `CommandWatchdog` (the Metal command timeout is a backend parameter),
  `Residency` for buffer residency, and `DeviceCapabilities` probing.
- `abi/` is the host/shader contract: one header per kernel family with the
  exact uniform layout and quant formats both sides agree on.
- `kernels/` — production sources grouped by execution phase: `prefill/`,
  `decode/`, `shared/`; templates both phases instantiate live in
  `common/`. Every `.metal` file compiles into the metallib at build time;
  kernels ship precompiled.

### ane/ — Neural Engine

Prefill FFN offload on the private `AppleNeuralEngine` framework: `Program`
compiles and runs the ANE program, `Handoff` transfers a Metal command's
work to it, `ProgramInstrumentation` measures it. `AneFfnCalibration` and
`AneFfnMeasurement` in `ops/` decide when the ANE path wins. Disabled with
`--disable-ane`; on by default for the 27B target.

## install/ (Python)

Installs models and wires up clients; runs as scripts with a PEP 366 header
so siblings import as the `install` package.

- `upstream.py` — the only module that asks the Hub: inspects the target
  repository, pairs the trained DFlash 2 draft, decides when to follow
  revisions.
- `hub.py` — sources, the Hub cache, and pinning (`refs/splash/...`) so
  pruning never removes files an installed model links.
- `assembly.py` — the assembly layout: local directories of links to pinned
  snapshots, built and verified, published atomically, with garbage
  collection and the GGUF-derived metadata (`model.json`).
- `families.py` — each model family's draft repository.
- `models.py` — model IDs, selections, the installation lock, the
  `prepare|verify|link` command line, and running the engine's checks.
- `gguf.py`, `paths.py`, `clients.py`, `launcher.py`, `completions/` —
  GGUF format validation (refusing all-BF16/F16/F32 targets), paths, agent
  client configuration, the CLI entry, shell completions.

An installation is immutable once published: revisions resolve to commits
once, and a new assembly replaces the old one atomically. Running servers
hold the assembly while they run, so a concurrent install cannot collect it.

## dev/ (tooling)

- `tests/` — `engine/` (native tests incl. the protocol golden), `install/`,
  `server/`, `tools/`, `fixtures/` (weight goldens), plus real-model checks
  (`agent_real.py`, `smoke_real.py`) and the native test peer.
- `benchmarks/` — A/B benchmark harness (`abba.py`), native decode/prefill
  and kernel benchmarks (`.mm`), HTTP and backend regressions, weight
  digests, and notes (`kv-formats.md`, `upstream-loading.md`).
- `tools/` — `check_architecture.py` (dependency boundaries, AwakeClock
  policy, no HF tokens in sources), build and kernel identity, packaging
  (`package.py`, `package_brew.py`), publishing checks, vision reference,
  MLX quantization fixture, `install.sh`.

## Build

The root `Makefile` drives everything with `xcrun clang++` and the Metal
toolchain; `dev/Makefile` and `dev/native.mk` hold the test and native
pieces. Key targets: `check-source` (architecture-check + source checks),
`check-native-cpu`, `check-native-build`, sanitizers, engine tests, and
release packaging. The `.venv` is created and stamped against
`install/requirements.txt`; `BuildIdentity.hpp` and the kernel identity are
generated into `build/` so the binary and its metallib always match. CI
(`.github/workflows/ci.yml`) runs the same source and native gates on
macOS, plus an opt-in real-model release gate.

## Cross-cutting invariants

- **One binary protocol, pinned bytes.** Python and C++ agree on
  `dev/tests/engine/protocol_golden.txt`; the protocol version and status
  schema version are constants in both languages.
- **In-memory weights, no on-disk conversion.** Sources load at disk speed
  straight into kernel layouts; the page cache holds no second copy.
- **AwakeClock.** Production code measures durations on
  `runtime/AwakeClock.hpp` (sleep-excluded), wall instants on
  `system_clock`; the architecture check forbids steady clocks and timed
  waits.
- **Layered imports.** `make architecture-check` keeps lower layers from
  importing the HTTP entry module, keeps `server/` and `install/` to
  relative package imports, keeps workspace sizing in `ops/`, and keeps the
  engine free of concrete models and client-specific behavior.
- **Deterministic artifacts.** Weight golden hashes, kernel identity, and
  the build identity make "what this binary runs" reproducible and checked.

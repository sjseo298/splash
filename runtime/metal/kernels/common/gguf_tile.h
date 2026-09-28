#pragma once
#include "metal/abi/Gguf.h"
#include <metal_stdlib>
using namespace metal;

// What both GGUF GEMM families (kernels/shared/gguf_linear.metal and
// kernels/decode/linear_gguf_sgmatrix.metal) do to a finished fp32 sum.
enum GgufEpilogue : ushort { EpNone = GGUF_EPILOGUE_NONE, EpResidual = GGUF_EPILOGUE_RESIDUAL, EpUpWithGate = GGUF_EPILOGUE_UP_WITH_GATE };

inline float gguf_silu(float g) { return g / (1.0f + fast::exp2(-1.44269504089f * g)); }

// The output of sum v at element `at`: v, v plus the residual aux[at], or the bf16-rounded up value v times silu of
// the gate aux[at], as the destination's type Out: bf16, or fp32 (the plain epilogue's logits).
template <GgufEpilogue Ep, class Out = bfloat> inline Out gguf_epilogue(float v, device const bfloat *aux, ulong at) {
  if constexpr (Ep == EpResidual) v += float(aux[at]);
  if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * gguf_silu(float(aux[at]));
  return Out(v);
}

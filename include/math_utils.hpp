#pragma once

#include <cstdint>

// nvcc defines __CUDACC__ for every translation unit it processes, host .cpp
// files included, so this only takes the plain-macro path in a host-only
// build (g++/clang, no CUDA toolkit) such as the unit test suite.
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

// Shared by every CPU reference / CUDA kernel pair that needs to go from an
// accumulated float back to a stored byte. Defined once, callable from both
// host and device code, so CPU and CUDA can't silently pick different
// rounding behavior (truncate vs. round) the way BT.709 vs. BT.601 weights
// once drifted apart between grayscale's two implementations.
__host__ __device__ inline uint8_t roundToByte(float value) {
    return static_cast<uint8_t>(value + 0.5f);
}

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Shared by every CPU reference / CUDA kernel pair that needs to go from an
// accumulated float back to a stored byte. Defined once, callable from both
// host and device code, so CPU and CUDA can't silently pick different
// rounding behavior (truncate vs. round) the way BT.709 vs. BT.601 weights
// once drifted apart between grayscale's two implementations.
__host__ __device__ inline uint8_t roundToByte(float value) {
    return static_cast<uint8_t>(value + 0.5f);
}

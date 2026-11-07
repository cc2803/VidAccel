#pragma once

#include "image.hpp"
#include <cassert>
#include <cstdlib>
#include <algorithm>

// Largest per-byte absolute difference between two HostImages of matching
// shape. The shared correctness check for every CPU-vs-CUDA kernel pair.
inline int maxAbsDiff(const HostImage& a, const HostImage& b) {
    assert(a.width == b.width && a.height == b.height && a.channels == b.channels);

    int maxDiff = 0;
    size_t n = a.bytes();
    for (size_t i = 0; i < n; ++i) {
        int diff = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

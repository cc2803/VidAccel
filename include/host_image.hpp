#pragma once

#include <cstdint>
#include <cstddef>

// Split out of image.hpp so CPU-only code (cpu_reference.cpp, verify.hpp,
// and the test suite) can be compiled with a plain host compiler and no
// CUDA toolkit. DeviceImage and the cudaMemcpy helpers stay in image.hpp,
// which this file is included by.
struct HostImage {
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t* data = nullptr;

    HostImage() = default;

    HostImage(int w, int h, int c) : width(w), height(h), channels(c) {
        data = new uint8_t[static_cast<size_t>(w) * h * c];
    }

    ~HostImage() {
        delete[] data;
    }

    HostImage(const HostImage&) = delete;
    HostImage& operator=(const HostImage&) = delete;

    // Default member-wise move would copy `data` without nulling the source,
    // so both objects' destructors would delete[] the same pointer. Null it
    // out explicitly to transfer ownership instead of sharing it.
    HostImage(HostImage&& other) noexcept
        : width(other.width), height(other.height), channels(other.channels), data(other.data) {
        other.data = nullptr;
        other.width = other.height = other.channels = 0;
    }

    HostImage& operator=(HostImage&& other) noexcept {
        if (this != &other) {
            delete[] data;
            width = other.width;
            height = other.height;
            channels = other.channels;
            data = other.data;
            other.data = nullptr;
            other.width = other.height = other.channels = 0;
        }
        return *this;
    }

    size_t bytes() const {
        return static_cast<size_t>(width) * height * channels;
    }
};

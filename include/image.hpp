#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include "host_image.hpp"

struct DeviceImage {
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t* data = nullptr;

    DeviceImage() = default;

    DeviceImage(int w, int h, int c) : width(w), height(h), channels(c) {
        cudaError_t err = cudaMalloc(&data, static_cast<size_t>(w) * h * c);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    ~DeviceImage() {
        if (data) {
            cudaFree(data);
        }
    }

    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;

    DeviceImage(DeviceImage&& other) noexcept
        : width(other.width), height(other.height), channels(other.channels), data(other.data) {
        other.data = nullptr;
        other.width = other.height = other.channels = 0;
    }

    DeviceImage& operator=(DeviceImage&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);
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

inline void copyToDevice(const HostImage& src, DeviceImage& dst) {
    cudaMemcpy(dst.data, src.data, src.bytes(), cudaMemcpyHostToDevice);
}

inline void copyToHost(const DeviceImage& src, HostImage& dst) {
    cudaMemcpy(dst.data, src.data, src.bytes(), cudaMemcpyDeviceToHost);
}

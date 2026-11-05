#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>

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

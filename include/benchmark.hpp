#pragma once

#include <cuda_runtime.h>
#include <chrono>

// Kernel-only timing (Rule 5: kernel latency, not H<->D transfer or CPU-side
// launch overhead). Caller is responsible for calling cudaDeviceSynchronize()
// or letting stop()'s cudaEventSynchronize do it before reading elapsedMs().
struct GpuTimer {
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~GpuTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start() {
        cudaEventRecord(startEvent);
    }

    void stop() {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
    }

    float elapsedMs() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        return ms;
    }
};

struct CpuTimer {
    std::chrono::high_resolution_clock::time_point startTime;

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - startTime).count();
    }
};

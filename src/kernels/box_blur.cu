#include <cstdio>
#include "image.hpp"
#include "math_utils.hpp"
#include "cpu_reference.hpp"
#include "verify.hpp"
#include "benchmark.hpp"

__global__ void boxBlurKernel(const uint8_t* rgb, uint8_t* blurred, int width, int height, int radius) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }

    int sumR = 0, sumG = 0, sumB = 0;
    int area = (2 * radius + 1) * (2 * radius + 1);

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {

            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);

            int rgbIdx = (ny * width + nx) * 3;
            sumR += rgb[rgbIdx];
            sumG += rgb[rgbIdx + 1];
            sumB += rgb[rgbIdx + 2];
        }
    }

    int blurredIdx = (y * width + x) * 3;
    blurred[blurredIdx] = roundToByte(static_cast<float>(sumR) / area);
    blurred[blurredIdx + 1] = roundToByte(static_cast<float>(sumG) / area);
    blurred[blurredIdx + 2] = roundToByte(static_cast<float>(sumB) / area);

}

// Single bright pixel on a dark background. Lets you hand-compute the
// expected spread instead of eyeballing it. (px, py) deliberately not fixed
// to the image center here — call this once centered and once near a
// border/corner so your test actually exercises the clamp path in #1, not
// just the interior case.
void makeImpulseImage(HostImage& img, int px, int py) {
    for (int i = 0; i < img.height * img.width * img.channels; ++i) {
        img.data[i] = 0;
    }
    int idx = (py * img.width + px) * img.channels;
    for (int c = 0; c < img.channels; ++c) {
        img.data[idx + c] = 255;
    }
}

void runBenchmark(int width, int height, int radius) {
    HostImage hostRgb(width, height, 3);
    makeImpulseImage(hostRgb, width / 2, height / 2);

    DeviceImage deviceRgb(width, height, 3);
    DeviceImage deviceBlurred(width, height, 3);
    copyToDevice(hostRgb, deviceRgb);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    GpuTimer gpuTimer;
    gpuTimer.start();
    boxBlurKernel<<<gridDim, blockDim>>>(deviceRgb.data, deviceBlurred.data, width, height, radius);
    gpuTimer.stop();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    HostImage cudaBlurred(width, height, 3);
    copyToHost(deviceBlurred, cudaBlurred);

    HostImage cpuBlurred(width, height, 3);
    CpuTimer cpuTimer;
    cpuTimer.start();
    boxBlur(hostRgb, cpuBlurred, radius);
    double cpuMs = cpuTimer.elapsedMs();

    int diff = maxAbsDiff(cudaBlurred, cpuBlurred);

    printf("\n[Benchmark %dx%d, radius=%d (%dx%d window)]\n",
           width, height, radius, 2 * radius + 1, 2 * radius + 1);
    printf("  CPU:  %.4f ms\n", cpuMs);
    printf("  CUDA: %.4f ms (kernel only)\n", gpuTimer.elapsedMs());
    printf("  Speedup (CPU/CUDA): %.2fx\n", cpuMs / gpuTimer.elapsedMs());
    printf("  Correctness: max abs diff %d %s\n", diff, diff <= 1 ? "(PASS)" : "(FAIL)");
}

void checkCorrectness(int width, int height, int radius, int px, int py, const char* label) {
    HostImage hostRgb(width, height, 3);
    makeImpulseImage(hostRgb, px, py);

    DeviceImage deviceRgb(width, height, 3);
    DeviceImage deviceBlurred(width, height, 3);
    copyToDevice(hostRgb, deviceRgb);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    boxBlurKernel<<<gridDim, blockDim>>>(deviceRgb.data, deviceBlurred.data, width, height, radius);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();

    HostImage cudaBlurred(width, height, 3);
    copyToHost(deviceBlurred, cudaBlurred);

    HostImage cpuBlurred(width, height, 3);
    boxBlur(hostRgb, cpuBlurred, radius);

    int diff = maxAbsDiff(cudaBlurred, cpuBlurred);
    printf("[%s] radius=%d, impulse at (%d,%d): max abs diff %d %s\n",
           label, radius, px, py, diff, diff <= 1 ? "(PASS)" : "(FAIL)");
}

int main() {
    const int width = 16;
    const int height = 16;
    const int radii[] = {1, 2, 5}; // 3x3, 5x5, 11x11

    for (int radius : radii) {
        checkCorrectness(width, height, radius, width / 2, height / 2, "interior");
        checkCorrectness(width, height, radius, 0, 0, "corner");
    }

    for (int radius : radii) {
        runBenchmark(16, 16, radius);
        runBenchmark(1920, 1080, radius);
    }

    return 0;
}

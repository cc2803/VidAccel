#include <cstdio>
#include "image.hpp"
#include "constants.hpp"
#include "cpu_reference.hpp"
#include "verify.hpp"

__global__ void grayscaleKernel(const uint8_t* rgb, uint8_t* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int rgbIdx = (y*width + x)*3;
    int grayIdx = y*width + x;

    int r = rgb[rgbIdx];
    int g = rgb[rgbIdx + 1];
    int b = rgb[rgbIdx + 2];
    int grayValue = R_WEIGHT * r + G_WEIGHT * g + B_WEIGHT * b;

    gray[grayIdx] = static_cast<uint8_t>(grayValue);
}

void makeSyntheticImage(HostImage& img) {
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            int idx = (y * img.width + x) * img.channels;
            img.data[idx + 0] = static_cast<uint8_t>(x * 16);
            img.data[idx + 1] = static_cast<uint8_t>(y * 16);
            img.data[idx + 2] = 128;
        }
    }
}

int main() {
    const int width = 16;
    const int height = 16;

    HostImage hostRgb(width, height, 3);
    makeSyntheticImage(hostRgb);

    DeviceImage deviceRgb(width, height, 3);
    DeviceImage deviceGray(width, height, 1);
    copyToDevice(hostRgb, deviceRgb);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    grayscaleKernel<<<gridDim, blockDim>>>(deviceRgb.data, deviceGray.data, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    HostImage hostGray(width, height, 1);
    copyToHost(deviceGray, hostGray);

    printf("Pixel (0,0): RGB=(%d,%d,%d) -> Gray=%d (expect ~8)\n",
           hostRgb.data[0], hostRgb.data[1], hostRgb.data[2], hostGray.data[0]);

    int idx15 = (0 * width + 15) * 3;
    int gidx15 = 0 * width + 15;
    printf("Pixel (15,0): RGB=(%d,%d,%d) -> Gray=%d\n",
           hostRgb.data[idx15], hostRgb.data[idx15 + 1], hostRgb.data[idx15 + 2], hostGray.data[gidx15]);

    HostImage cpuGray(width, height, 1);
    toGrayScale(hostRgb, cpuGray);

    int diff = maxAbsDiff(hostGray, cpuGray);
    printf("Max abs diff (CUDA vs CPU), full %dx%d image: %d %s\n",
           width, height, diff, diff <= 1 ? "(PASS)" : "(FAIL)");

    return 0;
}

#include "image.hpp"
#include "constants.hpp"
#include "math_utils.hpp"
#include <cassert>
#include <algorithm>

void toGrayScale(const HostImage& rgb, HostImage& gray){

    // to check if the input image is RGB
    assert(rgb.channels == 3);

    // to check if the output image is grayscale
    assert(gray.width == rgb.width && gray.height == rgb.height);
    assert(gray.channels == 1);
    
    for(int i = 0; i < rgb.height; i++){
        for(int j = 0; j < rgb.width; j++){
            int idx = (i * rgb.width + j) * rgb.channels;
            uint8_t r = rgb.data[idx];
            uint8_t g = rgb.data[idx + 1];
            uint8_t b = rgb.data[idx + 2];
            uint8_t grayValue = static_cast<uint8_t>(R_WEIGHT * r + G_WEIGHT * g + B_WEIGHT * b);
            gray.data[i * gray.width + j] = grayValue;
        }
    }
}

void boxBlur(const HostImage& rgb, HostImage& blurred, int radius){

    assert(rgb.channels == 3);
    assert(blurred.width == rgb.width && blurred.height == rgb.height);
    assert(blurred.channels == 3);

    int area = (2 * radius + 1) * (2 * radius + 1);

    for(int i = 0; i < rgb.height; i++){
        for(int j = 0; j < rgb.width; j++){
            int sumR = 0, sumG = 0, sumB = 0;

            for(int di = -radius; di <= radius; di++){
                for(int dj = -radius; dj <= radius; dj++){
                    int ni = std::min(std::max(i + di, 0), rgb.height - 1);
                    int nj = std::min(std::max(j + dj, 0), rgb.width - 1);
                    int idx = (ni * rgb.width + nj) * rgb.channels;
                    sumR += rgb.data[idx];
                    sumG += rgb.data[idx + 1];
                    sumB += rgb.data[idx + 2];
                }
            }

            int idxBlurred = (i * blurred.width + j) * blurred.channels;
            blurred.data[idxBlurred] = roundToByte(static_cast<float>(sumR) / area);
            blurred.data[idxBlurred + 1] = roundToByte(static_cast<float>(sumG) / area);
            blurred.data[idxBlurred + 2] = roundToByte(static_cast<float>(sumB) / area);
        }
    }

}
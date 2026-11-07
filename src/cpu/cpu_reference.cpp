#include "image.hpp"
#include "constants.hpp"
#include <cassert>

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
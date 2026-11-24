#pragma once

#include "host_image.hpp"

void toGrayScale(const HostImage& rgb, HostImage& gray);

// radius: 1 -> 3x3, 2 -> 5x5, 5 -> 11x11.
void boxBlur(const HostImage& rgb, HostImage& blurred, int radius);

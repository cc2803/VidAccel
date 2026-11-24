#include <catch2/catch_test_macros.hpp>

#include "host_image.hpp"
#include "cpu_reference.hpp"

namespace {

// Writes one RGB pixel at (x, y) into an image already sized for it.
void setPixel(HostImage& img, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    int idx = (y * img.width + x) * img.channels;
    img.data[idx] = r;
    img.data[idx + 1] = g;
    img.data[idx + 2] = b;
}

} // namespace

TEST_CASE("toGrayScale matches known-good values from the validated hardware run", "[cpu_reference][grayscale]") {
    // Same two pixels as the printf sanity checks in src/kernels/grayscale.cu,
    // confirmed correct against CUDA on real hardware (Experiment #1,
    // docs/experiments.md): max abs diff 0 at 16x16 and 1920x1080.
    HostImage rgb(2, 1, 3);
    setPixel(rgb, 0, 0, 0, 0, 128);
    setPixel(rgb, 1, 0, 240, 0, 128);

    HostImage gray(2, 1, 1);
    toGrayScale(rgb, gray);

    REQUIRE(gray.data[0] == 8);
    REQUIRE(gray.data[1] == 59);
}

TEST_CASE("boxBlur on a flat image is a no-op", "[cpu_reference][boxblur]") {
    // Averaging a window of identical values returns that same value, at
    // any radius, with no boundary case to trip over.
    HostImage rgb(5, 5, 3);
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            setPixel(rgb, x, y, 100, 150, 200);
        }
    }

    SECTION("radius 1 (3x3)") {
        HostImage blurred(5, 5, 3);
        boxBlur(rgb, blurred, 1);
        for (size_t i = 0; i < blurred.bytes(); ++i) {
            REQUIRE(blurred.data[i] == rgb.data[i]);
        }
    }

    SECTION("radius 2 (5x5)") {
        HostImage blurred(5, 5, 3);
        boxBlur(rgb, blurred, 2);
        for (size_t i = 0; i < blurred.bytes(); ++i) {
            REQUIRE(blurred.data[i] == rgb.data[i]);
        }
    }
}

TEST_CASE("boxBlur spreads a single impulse pixel by hand-computable amounts", "[cpu_reference][boxblur]") {
    // Impulse at the corner (0,0) on a 4x4 image, radius 1 (3x3 window,
    // area 9). Clamping makes (0,0) itself appear 4 times in its own
    // window (2 clamped row positions x 2 clamped col positions), so its
    // own blurred value isn't simply 255/9.
    HostImage rgb(4, 4, 3);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            setPixel(rgb, x, y, 0, 0, 0);
        }
    }
    setPixel(rgb, 0, 0, 255, 255, 255);

    HostImage blurred(4, 4, 3);
    boxBlur(rgb, blurred, 1);

    auto at = [&](int x, int y) { return blurred.data[(y * blurred.width + x) * blurred.channels]; };

    // (0,0)'s own window: impulse counted 4 times out of 9 -> round(4*255/9) = round(113.33) = 113.
    REQUIRE(at(0, 0) == 113);
    // (1,1)'s window fully covers (0,0) once, no clamping -> round(255/9) = round(28.33) = 28.
    REQUIRE(at(1, 1) == 28);
    // (3,3)'s window (clamped to rows/cols 2-3) never reaches (0,0) -> untouched.
    REQUIRE(at(3, 3) == 0);
}

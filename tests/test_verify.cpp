#include <catch2/catch_test_macros.hpp>

#include "host_image.hpp"
#include "verify.hpp"

TEST_CASE("maxAbsDiff on identical images returns 0", "[verify]") {
    HostImage a(2, 2, 3);
    HostImage b(2, 2, 3);
    for (size_t i = 0; i < a.bytes(); ++i) {
        a.data[i] = static_cast<uint8_t>(i * 7);
        b.data[i] = static_cast<uint8_t>(i * 7);
    }

    REQUIRE(maxAbsDiff(a, b) == 0);
}

TEST_CASE("maxAbsDiff finds the single differing byte", "[verify]") {
    HostImage a(2, 2, 1);
    HostImage b(2, 2, 1);
    for (size_t i = 0; i < a.bytes(); ++i) {
        a.data[i] = 50;
        b.data[i] = 50;
    }
    b.data[2] = 80; // |80 - 50| = 30

    REQUIRE(maxAbsDiff(a, b) == 30);
}

TEST_CASE("maxAbsDiff returns the largest diff, not the first or last one found", "[verify]") {
    HostImage a(1, 4, 1);
    HostImage b(1, 4, 1);
    for (size_t i = 0; i < a.bytes(); ++i) {
        a.data[i] = 0;
    }
    // Scattered diffs of increasing-then-decreasing size; the max (40) sits
    // in the middle, so this fails a "last seen wins" or "first seen wins" bug.
    b.data[0] = 5;
    b.data[1] = 40;
    b.data[2] = 10;
    b.data[3] = 1;

    REQUIRE(maxAbsDiff(a, b) == 40);
}

TEST_CASE("maxAbsDiff diffs in either direction count the same", "[verify]") {
    HostImage a(1, 2, 1);
    HostImage b(1, 2, 1);
    a.data[0] = 10;
    b.data[0] = 200; // b - a is large positive
    a.data[1] = 200;
    b.data[1] = 10; // b - a is large negative, same magnitude

    REQUIRE(maxAbsDiff(a, b) == 190);
}

cmake_minimum_required(VERSION 3.15)
project(VidAccel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This only builds the CPU-side correctness oracle (src/cpu/*.cpp) and its
# unit tests with a plain host compiler, no CUDA toolkit required. The CUDA
# kernels (src/kernels/*.cu) are built and run on Colab per docs/roadmap.md —
# there's no GPU on a CI runner to run them against, so CI's job is to catch
# regressions in the CPU oracle itself (the thing every kernel is checked
# against), not to build or execute device code.
add_library(vidaccel_core
    src/cpu/cpu_reference.cpp
)
target_include_directories(vidaccel_core PUBLIC include)

enable_testing()
add_subdirectory(tests)

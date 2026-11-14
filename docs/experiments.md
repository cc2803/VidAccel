# Experiment Journal

One entry per meaningful optimization attempt or benchmark — including ones that
regress or fail. Template:

```markdown
## Experiment #N — <title>

### Hypothesis

### Baseline

### Change

### Hardware

### Results

### Observation

### Conclusion

### Next experiment
```

---

## Experiment #1 — Grayscale: CPU vs. Naive CUDA

### Hypothesis

Grayscale should be **memory-bandwidth-bound, not compute-bound**:

- **Arithmetic cost:** 3 multiplies + 2 adds per pixel.
- **Memory traffic:** 3 bytes read + 1 byte written per pixel → ~1.25 FLOPs/byte,
  far below the point where a GPU's compute throughput would be the limiter.
- **Secondary prediction:** at a tiny resolution (16×16), CUDA may actually *lose*
  to the CPU — kernel launch overhead is a fixed cost, and there's almost no real
  work to hide it behind.

### Baseline

CPU reference: plain nested loop, [`src/cpu/cpu_reference.cpp`](../src/cpu/cpu_reference.cpp)
(`toGrayScale`). Both CPU and CUDA use identical BT.709 weights from
[`include/constants.hpp`](../include/constants.hpp), so any difference in output is a
real bug, not a formula mismatch.

### Change

First implementation — nothing optimized yet.

- **Mapping:** 1 thread → 1 output pixel.
- **Blocks:** 16×16, 2D.
- **Shared memory:** none — grayscale has no neighbor dependency, so there's nothing
  to tile.
- **Correctness gate:** `maxAbsDiff` (threshold ≤1) must pass before any timing
  number counts.

### Hardware

- **GPU:** NVIDIA Tesla T4 (Google Colab, free tier)
- **Driver:** 580.82.07
- **CUDA:** 13.0
- **VRAM:** 15360 MiB

### Results

| Resolution | CPU | CUDA (kernel only) | Speedup (CPU/CUDA) | Correctness |
|---|---:|---:|---:|---|
| 16×16 | 0.0025 ms | 0.0125 ms | **0.20×** | max abs diff 0 (PASS) |
| 1920×1080 | 20.4857 ms | 0.2127 ms | **96.33×** | max abs diff 0 (PASS) |

### Observation

**At 16×16, CUDA lost to the CPU by ~5x.** The image is only 256 pixels — almost no
real memory traffic or arithmetic — so the CUDA time is dominated by a roughly fixed
cost that exists regardless of input size: the CPU enqueuing the kernel with the
driver, and the GPU scheduling the grid/blocks before a single pixel is touched. The
CPU loop has none of that setup overhead; it just walks 256 iterations in cache.

**At 1920×1080 (~2M pixels, ~6.2MB moved), the result flips hard: CUDA wins by ~96x.**
The fixed launch cost becomes negligible next to the real work, and the GPU's actual
advantages take over — many threads moving bytes concurrently vs. the CPU's one core
doing it serially.

### Conclusion

The memory-bound hypothesis holds up directionally, but **the timing numbers alone
don't prove it** — they're equally consistent with "CPU is just slow at everything."
To actually confirm it rather than infer it, the next profiling pass needs:

- **Nsight Compute → Speed of Light (SOL) section**, specifically:
  - **Memory Throughput %** (of theoretical peak)
  - **Compute (SM) Throughput %** (of theoretical peak)
- A memory-bound kernel shows Memory Throughput sitting much closer to 100% than
  Compute Throughput — the kernel saturates the memory bus while the ALUs sit
  comparatively idle. If grayscale is truly memory-bound, Memory Throughput should
  dominate; if it doesn't, something else is the real bottleneck.

Deferred to when Nsight Compute profiling becomes the regular workflow (Phase 2/5) —
noted here as the specific check this experiment still owes.

### Next experiment

Box blur (3×3 → 5×5 → 11×11) — first op with a real neighbor dependency, sets up the
naive-vs-shared-memory comparison in Phase 2.

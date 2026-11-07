#pragma once

// ITU-R BT.709 luma weights (HD/digital video standard). Defined once so the
// CPU reference and CUDA kernel can never silently drift apart.
#define R_WEIGHT 0.21
#define G_WEIGHT 0.72
#define B_WEIGHT 0.07

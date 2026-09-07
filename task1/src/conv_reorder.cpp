// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            const float w = ker[ky * K + kx];
            const bool first = (ky == 0 && kx == 0);

            for (int oy = 0; oy < H; ++oy) {
                const float* row_in  = in + (oy + ky) * in_stride + kx;
                float*       row_out = out + oy * W;

                if (first) {
                    for (int ox = 0; ox < W; ++ox) row_out[ox] = row_in[ox] * w;
                } else {
                    for (int ox = 0; ox < W; ++ox) row_out[ox] += row_in[ox] * w;
                }
            }
        }
    }
    
}
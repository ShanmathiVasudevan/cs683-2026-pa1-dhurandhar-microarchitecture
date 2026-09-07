// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;
 
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 8) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
 
            for (int ky = 0; ky < K; ++ky) {
                const float* row_in  = in + (oy + ky) * in_stride + ox;
                const float* row_ker = ker + ky * K;
 
                for (int kx = 0; kx < K; ++kx) {
                    const float w = row_ker[kx];
                    acc0 += row_in[kx + 0] * w;
                    acc1 += row_in[kx + 1] * w;
                    acc2 += row_in[kx + 2] * w;
                    acc3 += row_in[kx + 3] * w;
                    acc4 += row_in[kx + 4] * w;
                    acc5 += row_in[kx + 5] * w;
                    acc6 += row_in[kx + 6] * w;
                    acc7 += row_in[kx + 7] * w;
                }
            }
 
            float* row_out = out + oy * W + ox;
            row_out[0] = acc0; row_out[1] = acc1; row_out[2] = acc2; row_out[3] = acc3;
            row_out[4] = acc4; row_out[5] = acc5; row_out[6] = acc6; row_out[7] = acc7;
        }
    }
    
}
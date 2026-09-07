// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
        const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int TILE_H = 32;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
        int oy_end = oy0 + TILE_H;
        if (oy_end > H) oy_end = H;

        for (int oy = oy0; oy < oy_end; ++oy) {
            float* out_row = out + oy * W;

            int ox = 0;
            for (; ox + 16 <= W; ox += 16) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();

                for (int ky = 0; ky < K; ++ky) {
                    const float* row_in = in + (oy + ky) * in_stride + ox;
                    for (int kx = 0; kx < K; ++kx) {
                        __m256 w  = _mm256_set1_ps(ker[ky * K + kx]);
                        __m256 v0 = _mm256_loadu_ps(row_in + kx);
                        __m256 v1 = _mm256_loadu_ps(row_in + kx + 8);
                        acc0 = _mm256_fmadd_ps(v0, w, acc0);
                        acc1 = _mm256_fmadd_ps(v1, w, acc1);
                    }
                }
                _mm256_storeu_ps(out_row + ox,     acc0);
                _mm256_storeu_ps(out_row + ox + 8, acc1);
            }

            for (; ox < W; ox += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (int ky = 0; ky < K; ++ky) {
                    const float* row_in = in + (oy + ky) * in_stride + ox;
                    for (int kx = 0; kx < K; ++kx) {
                        __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                        __m256 v = _mm256_loadu_ps(row_in + kx);
                        acc = _mm256_fmadd_ps(v, w, acc);
                    }
                }
                _mm256_storeu_ps(out_row + ox, acc);
            }
        }
    }
}
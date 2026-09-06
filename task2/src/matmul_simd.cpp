// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

static inline float hsum128_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    bool FLAG_128 = true;
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            if (FLAG_128){
                __m128 acc = _mm_setzero_ps();
                int p = 0;
                for (; p + 4 < K; p+=4) {
                    acc = _mm_fmadd_ps(_mm_loadu_ps(a + p), _mm_loadu_ps(b + p), acc);
                }
                float sum = hsum128_ps(acc);
                for (; p < K; ++p) sum += a[p] * b[p];   // K % 4 leftover
                C[static_cast<long>(i) * ldc + j] = sum;
            }
            else{
                 __m256 acc = _mm256_setzero_ps();
                int p = 0;
                for (; p + 8 <= K; p += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
                float sum = hsum256_ps(acc);
                for (; p < K; ++p) sum += a[p] * b[p];   // K % 8 leftover
                C[(long)i * ldc + j] = sum;
            }
        }
    }
}

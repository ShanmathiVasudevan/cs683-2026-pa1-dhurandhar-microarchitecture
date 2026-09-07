// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"
#ifndef PF_DIST
#define PF_DIST 64          // prefetch distance, in floats, ahead in the K-loop
#endif
#ifndef PF_HINT_ID
#define PF_HINT_ID 0        // 0=T0  1=T1  2=T2  3=NTA   (cache fill level)
#endif
#if   PF_HINT_ID == 0
#define PF_HINT _MM_HINT_T0
#elif PF_HINT_ID == 1
#define PF_HINT _MM_HINT_T1
#elif PF_HINT_ID == 2
#define PF_HINT _MM_HINT_T2
#else
#define PF_HINT _MM_HINT_NTA
#endif
void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; ++p) {
                if (p + PF_DIST < K) {                       
                    _mm_prefetch((const char*)(a + p + PF_DIST), PF_HINT);
                    _mm_prefetch((const char*)(b + p + PF_DIST), PF_HINT);
                }
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}

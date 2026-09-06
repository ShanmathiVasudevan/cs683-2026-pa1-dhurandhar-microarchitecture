// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"
static constexpr int MC = 64;
static constexpr int NC = 128;
static constexpr int PF_ROWS = 8;
#define PF_HINT _MM_HINT_T0

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

// 2 rows of A x 4 cols of B per pass: each B load reused by 2 A rows instead
// of 1, roughly doubling arithmetic intensity on the B side vs. a 1x4 tile.
static inline void kernel2x4(const float* a0, const float* a1, bool has_row1,
                             const float* b0, const float* b1, const float* b2, const float* b3,
                             int K, float* out /* [8]: row0 in [0..3], row1 in [4..7] */) {
    __m256 acc00=_mm256_setzero_ps(), acc01=_mm256_setzero_ps(), acc02=_mm256_setzero_ps(), acc03=_mm256_setzero_ps();
    __m256 acc10=_mm256_setzero_ps(), acc11=_mm256_setzero_ps(), acc12=_mm256_setzero_ps(), acc13=_mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= K; p += 8) {
        __m256 vb0=_mm256_loadu_ps(b0+p), vb1=_mm256_loadu_ps(b1+p);
        __m256 vb2=_mm256_loadu_ps(b2+p), vb3=_mm256_loadu_ps(b3+p);
        __m256 va0=_mm256_loadu_ps(a0+p);
        acc00=_mm256_fmadd_ps(va0,vb0,acc00); acc01=_mm256_fmadd_ps(va0,vb1,acc01);
        acc02=_mm256_fmadd_ps(va0,vb2,acc02); acc03=_mm256_fmadd_ps(va0,vb3,acc03);
        if (has_row1) {
            __m256 va1=_mm256_loadu_ps(a1+p);
            acc10=_mm256_fmadd_ps(va1,vb0,acc10); acc11=_mm256_fmadd_ps(va1,vb1,acc11);
            acc12=_mm256_fmadd_ps(va1,vb2,acc12); acc13=_mm256_fmadd_ps(va1,vb3,acc13);
        }
    }
    out[0]=hsum256_ps(acc00); out[1]=hsum256_ps(acc01); out[2]=hsum256_ps(acc02); out[3]=hsum256_ps(acc03);
    for (; p < K; ++p) {                                // edge case: K % 8 leftover
        float av0 = a0[p];
        out[0]+=av0*b0[p]; out[1]+=av0*b1[p]; out[2]+=av0*b2[p]; out[3]+=av0*b3[p];
    }
    if (has_row1) {
        out[4]=hsum256_ps(acc10); out[5]=hsum256_ps(acc11); out[6]=hsum256_ps(acc12); out[7]=hsum256_ps(acc13);
        int p2 = K - (K % 8);
        for (; p2 < K; ++p2) {                          // edge case: K % 8 leftover, row 1
            float av1 = a1[p2];
            out[4]+=av1*b0[p2]; out[5]+=av1*b1[p2]; out[6]+=av1*b2[p2]; out[7]+=av1*b3[p2];
        }
    }
}

static inline float dot256(const float* a, const float* b, int K) {
    __m256 acc = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= K; p += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    float sum = hsum256_ps(acc);
    for (; p < K; ++p) sum += a[p] * b[p];
    return sum;
}
void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    for (int ic = 0; ic < M; ic += MC) {                // M-panel of A/C
        int ic_end = (ic + MC < M)?(ic + MC):M;              // edge case: M % MC
        for (int jc = 0; jc < N; jc += NC) {                    // N-panel of B
            int jc_end = (jc + NC < N)?(jc + NC): N;                  // edge case: N % NC
            int i = ic;
            for (; i < ic_end; i += 2) {
                bool has_row1 = (i + 1 < ic_end);          // edge case: M % 2 leftover row
                if (i + 2 < ic_end)
                    _mm_prefetch((const char*)(A + (long)(i + 2) * lda), PF_HINT);
                const float* a0 = A + (long)i * lda;
                const float* a1 = has_row1 ? A + (long)(i+1) * lda : nullptr;

                int j = jc;
                for (; j + 4 <= jc_end; j += 4) {
                    if (j + 4 + PF_ROWS < jc_end)
                        _mm_prefetch((const char*)(B + (long)(j + 4 + PF_ROWS) * ldb), PF_HINT);
                    const float* b0 = B + (long)(j+0)*ldb, *b1 = B + (long)(j+1)*ldb;
                    const float* b2 = B + (long)(j+2)*ldb, *b3 = B + (long)(j+3)*ldb;
                    float out[8];
                    kernel2x4(a0, a1, has_row1, b0, b1, b2, b3, K, out);
                    C[(long)i*ldc+j+0]=out[0]; C[(long)i*ldc+j+1]=out[1];
                    C[(long)i*ldc+j+2]=out[2]; C[(long)i*ldc+j+3]=out[3];
                    if (has_row1) {
                        C[(long)(i+1)*ldc+j+0]=out[4]; C[(long)(i+1)*ldc+j+1]=out[5];
                        C[(long)(i+1)*ldc+j+2]=out[6]; C[(long)(i+1)*ldc+j+3]=out[7];
                    }
                }
                for (; j < jc_end; ++j) {                   // edge case: N % 4 leftover columns
                    const float* b = B + (long)j * ldb;
                    C[(long)i*ldc+j] = dot256(a0, b, K);
                    if (has_row1) C[(long)(i+1)*ldc+j] = dot256(a1, b, K);
                }
            }
        }
    }
}

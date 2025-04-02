#include "wekua.h"

__kernel void gemm_nxn(
        __constant const wk *const restrict A,
        __constant const wk *const restrict B,

        __global wks *const restrict C,

        const ulong A_row_pitch,
        const ulong B_row_pitch,
        const ulong C_row_pitch,

        const ulong cols

        #if HAS_ALPHA
        , const wks alpha
        #if HAS_BETA
        , const wks beta
        #endif
        #endif
) {
    const ulong i = get_global_id(0) << STRIDE;
    const ulong j = get_global_id(1) << STRIDE;

    private wk A_tmp_buffer[BLOCK_SIZE * (BLOCK_SIZE / WK_VECTOR_WIDTH)] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    private wk B_tmp_buffer[BLOCK_SIZE * (BLOCK_SIZE / WK_VECTOR_WIDTH)] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    private wks C_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));

    ulong C_base = i * C_row_pitch + j;
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
        __attribute__((opencl_unroll_hint))
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
#if HAS_BETA
            C_tmp_buffer[y * BLOCK_SIZE + x] = beta * C[C_base + x];
#else
            C_tmp_buffer[y * BLOCK_SIZE + x] = C[C_base + x];
#endif
        }
        C_base += C_row_pitch;
    }

    for (ulong k = 0; k < cols; k += (BLOCK_SIZE / WK_VECTOR_WIDTH)) {
#if A_TRANS
#else
        ulong A_base = i * A_row_pitch + k;
        for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong x = 0; x < (BLOCK_SIZE / WK_VECTOR_WIDTH); x += 1) {
                A_tmp_buffer[y * BLOCK_SIZE + x] = A[A_base + x];
            }
            A_base += A_row_pitch;
        }
#endif

#if B_TRANS
        ulong B_base = j * B_row_pitch + k;
        for (ulong y = 0; y < (BLOCK_SIZE / WK_VECTOR_WIDTH); y += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
                B_tmp_buffer[y * BLOCK_SIZE + x] = B[B_base + x];
            }
            B_base += B_row_pitch;
        }
#else
#endif

        for (ulong y = 0; y < BLOCK_SIZE; y += 2) {
            for (ulong x = 0; x < BLOCK_SIZE; x += 2) {
#if WK_VECTOR_WIDTH == 1
                wk C11 = 0;
                wk C12 = 0;
                wk C21 = 0;
                wk C22 = 0;
#else
                wk C11 = (wk)(0);
                wk C12 = (wk)(0);
                wk C21 = (wk)(0);
                wk C22 = (wk)(0);
#endif

                for (ulong z = 0; z < BLOCK_SIZE / WK_VECTOR_WIDTH; z += 2) {
                    ulong base = y * (BLOCK_SIZE / WK_VECTOR_WIDTH) + z;
                    const wk A11 = A[base];
                    const wk A12 = A[base + 1];
                    const wk A21 = A[base + (BLOCK_SIZE / WK_VECTOR_WIDTH)];
                    const wk A22 = A[base + (BLOCK_SIZE / WK_VECTOR_WIDTH) + 1];

                    base = x * (BLOCK_SIZE / WK_VECTOR_WIDTH) + z;
                    const wk B11 = B[base];
                    const wk B21 = B[base];
                    const wk B12 = B[base + (BLOCK_SIZE / WK_VECTOR_WIDTH)];
                    const wk B22 = B[base + (BLOCK_SIZE / WK_VECTOR_WIDTH) + 1];

                    const wk s1 = (B12 - B22) * A11;
                    const wk s2 = (B21 - B11) * A22;
                    const wk s3 = (A11 + A12) * B22;
                    const wk s4 = (A21 + A22) * B11;
                    const wk s5 = (A11 + A22) * (B11 + B22);
                    const wk s6 = (A12 - A22) * (B21 + B22);
                    const wk s7 = (A11 - A21) * (B11 + B12);

                    C11 += s5 + s2 - s3 + s6;
                    C12 += s1 + s3;
                    C21 += s2 + s4;
                    C22 += s5 + s1 - s4 - s7;
                }

                const ulong base = y * BLOCK_SIZE + x;
#if WK_VECTOR_WIDTH == 1
                C[base] += C11;
                C[base + 1] += C12;
                C[base + BLOCK_SIZE] += C21;
                C[base + BLOCK_SIZE + 1] += C22;
#else
                C[base] += sum(C11);
                C[base + 1] += sum(C12);
                C[base + BLOCK_SIZE] += sum(C21);
                C[base + BLOCK_SIZE + 1] += sum(C22);
#endif
            }
        }

    }
}

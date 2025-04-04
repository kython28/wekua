#include "wekua.h"

__kernel void gemm(
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

    private wk A_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    private wk B_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    private wk C_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE))) = {0};

    for (ulong k = 0; k < cols; k += BLOCK_SIZE) {
        ulong base_index;
#if A_TRANS
        base_index = k * A_row_pitch + i;
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
                A_tmp_buffer[y * BLOCK_SIZE + x] = A[base_index + y];
            }
            base_index += A_row_pitch;
        }
#else
        base_index = i * A_row_pitch + k;
        for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
                A_tmp_buffer[y * BLOCK_SIZE + x] = A[base_index + x];
            }
            base_index += A_row_pitch;
        }
#endif

#if B_TRANS
        base_index = j * B_row_pitch + k;
        for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
                B_tmp_buffer[y * BLOCK_SIZE + x] = B[base_index + x];
            }
            base_index += B_row_pitch;
        }
#else
        base_index = k * B_row_pitch + j;
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
                B_tmp_buffer[y * BLOCK_SIZE + x] = B[base_index + y];
            }
            base_index += B_row_pitch;
        }
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

                __attribute__((opencl_unroll_hint))
                for (ulong z = 0; z < BLOCK_SIZE; z += 2) {
                    base_index = y * BLOCK_SIZE + z;
                    const wk A11 = A_tmp_buffer[base_index];
                    const wk A12 = A_tmp_buffer[base_index + 1];
                    const wk A21 = A_tmp_buffer[base_index + BLOCK_SIZE];
                    const wk A22 = A_tmp_buffer[base_index + BLOCK_SIZE + 1];

                    base_index = x * BLOCK_SIZE + z;
                    const wk B11 = B_tmp_buffer[base_index];
                    const wk B21 = B_tmp_buffer[base_index + 1];
                    const wk B12 = B_tmp_buffer[base_index + BLOCK_SIZE];
                    const wk B22 = B_tmp_buffer[base_index + BLOCK_SIZE + 1];

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

                base_index = y * BLOCK_SIZE + x;
                C_tmp_buffer[base_index] += C11;
                C_tmp_buffer[base_index + 1] += C12;
                C_tmp_buffer[base_index + BLOCK_SIZE] += C21;
                C_tmp_buffer[base_index + BLOCK_SIZE + 1] += C22;
            }
        }

    }

    ulong C_base = i * C_row_pitch + j;
    __attribute__((opencl_unroll_hint))
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
        __attribute__((opencl_unroll_hint))
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
#if WK_VECTOR_WIDTH == 1
#if HAS_ALPHA
#if HAS_BETA
            C[C_base + x] = alpha * C_tmp_buffer[y * BLOCK_SIZE + x] + beta * C[C_base + x];
#else
            C[C_base + x] = alpha * C_tmp_buffer[y * BLOCK_SIZE + x];
#endif
#else
            C[C_base + x] = C_tmp_buffer[y * BLOCK_SIZE + x];
#endif
#else
#if HAS_ALPHA
#if HAS_BETA
            C[C_base + x] = alpha * sum(C_tmp_buffer[y * BLOCK_SIZE + x]) + beta * C[C_base + x];
#else
            C[C_base + x] = alpha * sum(C_tmp_buffer[y * BLOCK_SIZE + x]);
#endif
#else
            C[C_base + x] = sum(C_tmp_buffer[y * BLOCK_SIZE + x]);
#endif
#endif
        }
        C_base += C_row_pitch;
    }
}

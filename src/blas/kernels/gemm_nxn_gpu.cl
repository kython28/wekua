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
    const ulong i = get_global_id(0) << 1;
    const ulong j = get_global_id(1) << 1;

    const ulong y = get_local_id(0) << 1;
    const ulong x = get_local_id(1) << 1;

    local wk A_tmp_buffer[4 * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    local wk B_tmp_buffer[4 * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));

#if A_TRANS == 0
    const ulong row_A = i*A_row_pitch;
#endif

#if B_TRANS
    const ulong row_B = j*B_row_pitch;
#endif

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

    const ulong local_base_index_A = y << 1;
    const ulong local_base_index_B = x << 1;
    for (ulong k = 0; k < cols; k += 2) {
        ulong base_index;

        private wk A11, A12, A21, A22;
        private wk B11, B12, B21, B22;

        if (x == 0) {
#if A_TRANS
            base_index = k * A_row_pitch + i;

            A11 = A[base_index];
            A21 = A[base_index + 1];

            A12 = A[base_index + A_row_pitch];
            A22 = A[base_index + A_row_pitch + 1];

            A_tmp_buffer[local_base_index_A] = A11;
            A_tmp_buffer[local_base_index_A + 1] = A12;
            A_tmp_buffer[local_base_index_A + 2] = A21;
            A_tmp_buffer[local_base_index_A + 3] = A22;
#else
            base_index = row_A + k;

            A11 = A[base_index];
            A12 = A[base_index + 1];

            A21 = A[base_index + A_row_pitch];
            A22 = A[base_index + A_row_pitch + 1];

            A_tmp_buffer[local_base_index_A] = A11;
            A_tmp_buffer[local_base_index_A + 1] = A12;
            A_tmp_buffer[local_base_index_A + 2] = A21;
            A_tmp_buffer[local_base_index_A + 3] = A22;

#endif
        }

        if (y == 0) {
#if B_TRANS
            base_index = row_B + k;
            
            B11 = B[base_index];
            B21 = B[base_index + 1];

            B12 = B[base_index + B_row_pitch];
            B22 = B[base_index + B_row_pitch + 1];

            B_tmp_buffer[local_base_index_B] = B11;
            B_tmp_buffer[local_base_index_B + 1] = B21;
            B_tmp_buffer[local_base_index_B + 2] = B12;
            B_tmp_buffer[local_base_index_B + 3] = B22;
#else
            base_index = k * B_row_pitch + j;

            B11 = B[base_index];
            B12 = B[base_index + 1];

            B21 = B[base_index + B_row_pitch];
            B22 = B[base_index + B_row_pitch + 1];

            B_tmp_buffer[local_base_index_B] = B11;
            B_tmp_buffer[local_base_index_B + 1] = B21;
            B_tmp_buffer[local_base_index_B + 2] = B12;
            B_tmp_buffer[local_base_index_B + 3] = B22;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (x != 0) {
            A11 = A_tmp_buffer[local_base_index_A];
            A12 = A_tmp_buffer[local_base_index_A + 1];
            A21 = A_tmp_buffer[local_base_index_A + 2];
            A22 = A_tmp_buffer[local_base_index_A + 3];
        }

        if (y != 0) {
            B11 = B_tmp_buffer[local_base_index_B];
            B21 = B_tmp_buffer[local_base_index_B + 1];
            B12 = B_tmp_buffer[local_base_index_B + 2];
            B22 = B_tmp_buffer[local_base_index_B + 3];
        }

        C11 = A11 * B11 + A12 * B21 + C11;
        C12 = A11 * B12 + A12 * B22 + C12;
        C21 = A21 * B11 + A22 * B21 + C21;
        C22 = A21 * B12 + A22 * B22 + C22;

        /* const wk s1 = (B12 - B22) * A11; */
        /* const wk s2 = (B21 - B11) * A22; */
        /* const wk s3 = (A11 + A12) * B22; */
        /* const wk s4 = (A21 + A22) * B11; */
        /* const wk s5 = (A11 + A22) * (B11 + B22); */
        /* const wk s6 = (A12 - A22) * (B21 + B22); */
        /* const wk s7 = (A11 - A21) * (B11 + B12); */

        /* C11 += s5 + s2 - s3 + s6; */
        /* C12 += s1 + s3; */
        /* C21 += s2 + s4; */
        /* C22 += s5 + s1 - s4 - s7; */
    }

    const ulong C_index = i*C_row_pitch + j;
    const ulong C_index2 = C_index + C_row_pitch;

#if WK_VECTOR_WIDTH == 1

#if HAS_ALPHA
#if HAS_BETA
    C[C_index] = alpha*C11 + beta*C[C_index];
    C[C_index + 1] = alpha*C12 + beta*C[C_index + 1];
    C[C_index2] = alpha*C21 + beta*C[C_index2];
    C[C_index2 + 1] = alpha*C22 + beta*C[C_index2 + 1];
#else
    C[C_index] = alpha*C11;
    C[C_index + 1] = alpha*C12;
    C[C_index2] = alpha*C21;
    C[C_index2 + 1] = alpha*C22;
#endif
#else
    C[C_index] = C11;
    C[C_index + 1] = C12;
    C[C_index2] = C21;
    C[C_index2 + 1] = C22;
#endif

#else

#if HAS_ALPHA
#if HAS_BETA
    C[C_index] = alpha*sum(C11) + beta*C[C_index];
    C[C_index + 1] = alpha*sum(C12) + beta*C[C_index + 1];
    C[C_index2] = alpha*sum(C21) + beta*C[C_index2];
    C[C_index2 + 1] = alpha*sum(C22) + beta*C[C_index2 + 1];
#else
    C[C_index] = alpha*sum(C11);
    C[C_index + 1] = alpha*sum(C12);
    C[C_index2] = alpha*sum(C21);
    C[C_index2 + 1] = alpha*sum(C22);
#endif
#else
    C[C_index] = sum(C11);
    C[C_index + 1] = sum(C12);
    C[C_index2] = sum(C21);
    C[C_index2 + 1] = sum(C22);
#endif

#endif
}

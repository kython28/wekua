#include "wekua.h"

#define FILL_TILE(tile, values, row_index, col_index, row_pitch) \
    base_index = row_index * row_pitch + col_index; \
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) { \
        __attribute__((opencl_unroll_hint)) \
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) { \
            tile[y * BLOCK_SIZE + x] = values[base_index + x]; \
        } \
        base_index += row_pitch; \
    }

#define FILL_TRANSPOSED_TILE(tile, values, row_index, col_index, row_pitch) \
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) { \
        base_index = row_index * row_pitch + col_index + y; \
        __attribute__((opencl_unroll_hint)) \
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) { \
            tile[y * BLOCK_SIZE + x] = values[base_index]; \
            base_index += row_pitch; \
        } \
    }

__kernel void gemm(
    __global const wk *const restrict A,
    __global const wk *const restrict B,

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
        FILL_TILE(A_tmp_buffer, A, i, k, A_row_pitch)
#else
        FILL_TRANSPOSED_TILE(A_tmp_buffer, A, i, k, A_row_pitch)
#endif

#if B_TRANS
        FILL_TRANSPOSED_TILE(B_tmp_buffer, B, j, k, B_row_pitch)
#else
        FILL_TILE(B_tmp_buffer, B, j, k, B_row_pitch)
#endif

#if WK_VECTOR_WIDTH <= 8
        for (ulong z = 0; z < BLOCK_SIZE; z += 2) {
            for (ulong y = 0; y < BLOCK_SIZE; y += 2) {

                base_index = y * BLOCK_SIZE + z;
                const wk A11 = A_tmp_buffer[base_index];
                const wk A21 = A_tmp_buffer[base_index + 1];
                const wk A12 = A_tmp_buffer[base_index + BLOCK_SIZE];
                const wk A22 = A_tmp_buffer[base_index + BLOCK_SIZE + 1];

                __attribute__((opencl_unroll_hint))
                for (ulong x = 0; x < BLOCK_SIZE; x += 2) {
                    base_index = x * BLOCK_SIZE + z;
                    const wk B11 = B_tmp_buffer[base_index];
                    const wk B12 = B_tmp_buffer[base_index + 1];
                    const wk B21 = B_tmp_buffer[base_index + BLOCK_SIZE];
                    const wk B22 = B_tmp_buffer[base_index + BLOCK_SIZE + 1];

                    base_index = y * BLOCK_SIZE + x;
                    C_tmp_buffer[base_index] += A11 * B11 + A12 * B21;
                    C_tmp_buffer[base_index + 1] += A11 * B12 + A12 * B22;
                    C_tmp_buffer[base_index + BLOCK_SIZE] += A21 * B11 + A22 * B21;
                    C_tmp_buffer[base_index + BLOCK_SIZE + 1] += A21 * B12 + A22 * B22;
                }
            }
        }
#else
        for (ulong z = 0; z < BLOCK_SIZE; z += 4) {
            for (ulong y = 0; y < BLOCK_SIZE; y += 4) {

                base_index = y * BLOCK_SIZE + z;
                const wk A11 = A_tmp_buffer[base_index];
                const wk A12 = A_tmp_buffer[base_index + 1];
                const wk A13 = A_tmp_buffer[base_index + 2];
                const wk A14 = A_tmp_buffer[base_index + 3];

                base_index += BLOCK_SIZE;
                const wk A21 = A_tmp_buffer[base_index];
                const wk A22 = A_tmp_buffer[base_index + 1];
                const wk A23 = A_tmp_buffer[base_index + 2];
                const wk A24 = A_tmp_buffer[base_index + 3];

                base_index += BLOCK_SIZE;
                const wk A31 = A_tmp_buffer[base_index];
                const wk A32 = A_tmp_buffer[base_index + 1];
                const wk A33 = A_tmp_buffer[base_index + 2];
                const wk A34 = A_tmp_buffer[base_index + 3];

                base_index += BLOCK_SIZE;
                const wk A41 = A_tmp_buffer[base_index];
                const wk A42 = A_tmp_buffer[base_index + 1];
                const wk A43 = A_tmp_buffer[base_index + 2];
                const wk A44 = A_tmp_buffer[base_index + 3];


                __attribute__((opencl_unroll_hint))
                for (ulong x = 0; x < BLOCK_SIZE; x += 4) {
                    base_index = x * BLOCK_SIZE + z;
                    const wk B11 = B_tmp_buffer[base_index];
                    const wk B21 = B_tmp_buffer[base_index + 1];
                    const wk B31 = B_tmp_buffer[base_index + 2];
                    const wk B41 = B_tmp_buffer[base_index + 3];

                    base_index += BLOCK_SIZE;
                    const wk B12 = B_tmp_buffer[base_index];
                    const wk B22 = B_tmp_buffer[base_index + 1];
                    const wk B32 = B_tmp_buffer[base_index + 2];
                    const wk B42 = B_tmp_buffer[base_index + 3];

                    base_index += BLOCK_SIZE;
                    const wk B13 = B_tmp_buffer[base_index];
                    const wk B23 = B_tmp_buffer[base_index + 1];
                    const wk B33 = B_tmp_buffer[base_index + 2];
                    const wk B43 = B_tmp_buffer[base_index + 3];

                    base_index += BLOCK_SIZE;
                    const wk B14 = B_tmp_buffer[base_index];
                    const wk B24 = B_tmp_buffer[base_index + 1];
                    const wk B34 = B_tmp_buffer[base_index + 2];
                    const wk B44 = B_tmp_buffer[base_index + 3];

                    base_index = y * BLOCK_SIZE + x;
                    C_tmp_buffer[base_index] += A11 * B11 + A12 * B21 + A13 * B31 + A14 * B41;
                    C_tmp_buffer[base_index + 1] += A11 * B12 + A12 * B22 + A13 * B32 + A14 * B42;
                    C_tmp_buffer[base_index + 2] += A11 * B13 + A12 * B23 + A13 * B33 + A14 * B43;
                    C_tmp_buffer[base_index + 3] += A11 * B14 + A12 * B24 + A13 * B34 + A14 * B44;

                    base_index += BLOCK_SIZE;
                    C_tmp_buffer[base_index] += A21 * B11 + A22 * B21 + A23 * B31 + A24 * B41;
                    C_tmp_buffer[base_index + 1] += A21 * B12 + A22 * B22 + A23 * B32 + A24 * B42;
                    C_tmp_buffer[base_index + 2] += A21 * B13 + A22 * B23 + A23 * B33 + A24 * B43;
                    C_tmp_buffer[base_index + 3] += A21 * B14 + A22 * B24 + A23 * B34 + A24 * B44;

                    base_index += BLOCK_SIZE;
                    C_tmp_buffer[base_index] += A31 * B11 + A32 * B21 + A33 * B31 + A34 * B41;
                    C_tmp_buffer[base_index + 1] += A31 * B12 + A32 * B22 + A33 * B32 + A34 * B42;
                    C_tmp_buffer[base_index + 2] += A31 * B13 + A32 * B23 + A33 * B33 + A34 * B43;
                    C_tmp_buffer[base_index + 3] += A31 * B14 + A32 * B24 + A33 * B34 + A34 * B44;

                    base_index += BLOCK_SIZE;
                    C_tmp_buffer[base_index] += A41 * B11 + A42 * B21 + A43 * B31 + A44 * B41;
                    C_tmp_buffer[base_index + 1] += A41 * B12 + A42 * B22 + A43 * B32 + A44 * B42;
                    C_tmp_buffer[base_index + 2] += A41 * B13 + A42 * B23 + A43 * B33 + A44 * B43;
                    C_tmp_buffer[base_index + 3] += A41 * B14 + A42 * B24 + A43 * B34 + A44 * B44;
                }

            }
        }
#endif
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

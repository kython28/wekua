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

#if WK_COMPLEX
    COMPLEX_MUL_K(T)
#endif

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

        for (ulong z = 0; z < BLOCK_SIZE; z += 1) {
            for (ulong y = 0; y < BLOCK_SIZE; y += 4) {

                base_index = z * BLOCK_SIZE + y;
                const wk A1 = A_tmp_buffer[base_index];
                const wk A2 = A_tmp_buffer[base_index + 1];
                const wk A3 = A_tmp_buffer[base_index + 2];
                const wk A4 = A_tmp_buffer[base_index + 3];

                __attribute__((opencl_unroll_hint))
                for (ulong x = 0; x < BLOCK_SIZE; x += 4) {
                    base_index = z * BLOCK_SIZE + x;
                    const wk B1 = B_tmp_buffer[base_index];
                    const wk B2 = B_tmp_buffer[base_index + 1];
                    const wk B3 = B_tmp_buffer[base_index + 2];
                    const wk B4 = B_tmp_buffer[base_index + 3];

                    base_index = y * BLOCK_SIZE + x;
#if WK_COMPLEX
                    wk prod, prev;

                    COMPLEX_MUL(A1, B1, prod); prev = C_tmp_buffer[base_index];
                    C_tmp_buffer[base_index] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A1, B2, prod); prev = C_tmp_buffer[base_index + 1];
                    C_tmp_buffer[base_index + 1] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A1, B3, prod); prev = C_tmp_buffer[base_index + 2];
                    C_tmp_buffer[base_index + 2] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A1, B4, prod); prev = C_tmp_buffer[base_index + 3];
                    C_tmp_buffer[base_index + 3] = {prev.real + prod.real, prev.imag + prod.imag};

                    base_index += BLOCK_SIZE;
                    COMPLEX_MUL(A2, B1, prod); prev = C_tmp_buffer[base_index];
                    C_tmp_buffer[base_index] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A2, B2, prod); prev = C_tmp_buffer[base_index + 1];
                    C_tmp_buffer[base_index + 1] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A2, B3, prod); prev = C_tmp_buffer[base_index + 2];
                    C_tmp_buffer[base_index + 2] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A2, B4, prod); prev = C_tmp_buffer[base_index + 3];
                    C_tmp_buffer[base_index + 3] = {prev.real + prod.real, prev.imag + prod.imag};

                    base_index += BLOCK_SIZE;
                    COMPLEX_MUL(A3, B1, prod); prev = C_tmp_buffer[base_index];
                    C_tmp_buffer[base_index] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A3, B2, prod); prev = C_tmp_buffer[base_index + 1];
                    C_tmp_buffer[base_index + 1] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A3, B3, prod); prev = C_tmp_buffer[base_index + 2];
                    C_tmp_buffer[base_index + 2] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A3, B4, prod); prev = C_tmp_buffer[base_index + 3];
                    C_tmp_buffer[base_index + 3] = {prev.real + prod.real, prev.imag + prod.imag};

                    base_index += BLOCK_SIZE;
                    COMPLEX_MUL(A4, B1, prod); prev = C_tmp_buffer[base_index];
                    C_tmp_buffer[base_index] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A4, B2, prod); prev = C_tmp_buffer[base_index + 1];
                    C_tmp_buffer[base_index + 1] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A4, B3, prod); prev = C_tmp_buffer[base_index + 2];
                    C_tmp_buffer[base_index + 2] = {prev.real + prod.real, prev.imag + prod.imag};
                    COMPLEX_MUL(A4, B4, prod); prev = C_tmp_buffer[base_index + 3];
                    C_tmp_buffer[base_index + 3] = {prev.real + prod.real, prev.imag + prod.imag};
#else
                    C_tmp_buffer[base_index] += A1 * B1;
                    C_tmp_buffer[base_index + 1] += A1 * B2;
                    C_tmp_buffer[base_index + 2] += A1 * B3;
                    C_tmp_buffer[base_index + 3] += A1 * B4;

                    base_index += BLOCK_SIZE;
                    C_tmp_buffer[base_index] += A2 * B1;
                    C_tmp_buffer[base_index + 1] += A2 * B2;
                    C_tmp_buffer[base_index + 2] += A2 * B3;
                    C_tmp_buffer[base_index + 3] += A2 * B4;

                    base_index += BLOCK_SIZE;
                    C_tmp_buffer[base_index] += A3 * B1;
                    C_tmp_buffer[base_index + 1] += A3 * B2;
                    C_tmp_buffer[base_index + 2] += A3 * B3;
                    C_tmp_buffer[base_index + 3] += A3 * B4;

                    base_index += BLOCK_SIZE;
                    C_tmp_buffer[base_index] += A4 * B1;
                    C_tmp_buffer[base_index + 1] += A4 * B2;
                    C_tmp_buffer[base_index + 2] += A4 * B3;
                    C_tmp_buffer[base_index + 3] += A4 * B4;
#endif
                }
            }
        }
    }

    ulong C_base = i * C_row_pitch + j;
    __attribute__((opencl_unroll_hint))
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
        __attribute__((opencl_unroll_hint))
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
#if WK_COMPLEX
#if HAS_ALPHA
            wk tmp_val = C_tmp_buffer[y * BLOCK_SIZE + x];
            wk scaled;
            COMPLEX_MUL(tmp_val, alpha, scaled);
#if HAS_BETA
            wk old_val = C[C_base + x];
            wk beta_scaled;
            COMPLEX_MUL(old_val, beta, beta_scaled);
            C[C_base + x] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };
#else
            C[C_base + x] = scaled;
#endif
#else
            C[C_base + x] = C_tmp_buffer[y * BLOCK_SIZE + x];
#endif
#elif WK_VECTOR_WIDTH == 1
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

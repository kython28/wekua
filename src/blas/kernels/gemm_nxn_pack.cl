#include "wekua.h"

__kernel void gemm(
    __global const wk *const restrict A_packed,
    __global const wk *const restrict B_packed,

    __global wks *const restrict C,

    const ulong A_slice_pitch,
    const ulong A_row_pitch,

    const ulong B_slice_pitch,
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
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    const ulong C_row = i << STRIDE;
    const ulong C_col = j << STRIDE;

    private wk A_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    private wk B_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    private wk C_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE))) = {0};

    const ulong A_base_index = i * A_slice_pitch;
    const ulong B_base_index = j * B_slice_pitch;

#if WK_COMPLEX

    COMPLEX_MUL_K(T)

    for (ulong k = 0; k < cols; k += 1) {
        ulong base_index = A_base_index + k * A_row_pitch;
        ulong tile_index = 0;

        for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
                A_tmp_buffer[tile_index + x] = A_packed[base_index + x];
            }
            base_index += BLOCK_SIZE;
            tile_index += BLOCK_SIZE;
        }

        base_index = B_base_index + k * B_row_pitch;
        tile_index = 0;
        for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
                B_tmp_buffer[tile_index + x] = B_packed[base_index + x];
            }
            base_index += BLOCK_SIZE;
            tile_index += BLOCK_SIZE;
        }

        for (ulong y = 0; y < BLOCK_SIZE; y += 2) {
            for (ulong x = 0; x < BLOCK_SIZE; x += 2) {
                wk C11 = {0, 0};
                wk C12 = {0, 0};
                wk C21 = {0, 0};
                wk C22 = {0, 0};

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

                    wk prod;

                    COMPLEX_MUL(A11, B11, prod);
                    C11.real += prod.real; C11.imag += prod.imag;
                    COMPLEX_MUL(A12, B21, prod);
                    C11.real += prod.real; C11.imag += prod.imag;

                    COMPLEX_MUL(A11, B12, prod);
                    C12.real += prod.real; C12.imag += prod.imag;
                    COMPLEX_MUL(A12, B22, prod);
                    C12.real += prod.real; C12.imag += prod.imag;

                    COMPLEX_MUL(A21, B11, prod);
                    C21.real += prod.real; C21.imag += prod.imag;
                    COMPLEX_MUL(A22, B21, prod);
                    C21.real += prod.real; C21.imag += prod.imag;

                    COMPLEX_MUL(A21, B12, prod);
                    C22.real += prod.real; C22.imag += prod.imag;
                    COMPLEX_MUL(A22, B22, prod);
                    C22.real += prod.real; C22.imag += prod.imag;
                }

                base_index = y * BLOCK_SIZE + x;
                C_tmp_buffer[base_index].real += C11.real;
                C_tmp_buffer[base_index].imag += C11.imag;
                C_tmp_buffer[base_index + 1].real += C12.real;
                C_tmp_buffer[base_index + 1].imag += C12.imag;
                C_tmp_buffer[base_index + BLOCK_SIZE].real += C21.real;
                C_tmp_buffer[base_index + BLOCK_SIZE].imag += C21.imag;
                C_tmp_buffer[base_index + BLOCK_SIZE + 1].real += C22.real;
                C_tmp_buffer[base_index + BLOCK_SIZE + 1].imag += C22.imag;
            }
        }
    }

    ulong C_base = C_row * C_row_pitch + C_col;
    __attribute__((opencl_unroll_hint))
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
        __attribute__((opencl_unroll_hint))
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
#if HAS_ALPHA
#if HAS_BETA
            wk tmp_val = C_tmp_buffer[y * BLOCK_SIZE + x];
            wk scaled;
            COMPLEX_MUL(tmp_val, alpha, scaled);
            wk old_val = C[C_base + x];
            wk beta_scaled;
            COMPLEX_MUL(old_val, beta, beta_scaled);
            C[C_base + x] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };
#else
            wk tmp_val = C_tmp_buffer[y * BLOCK_SIZE + x];
            wk scaled;
            COMPLEX_MUL(tmp_val, alpha, scaled);
            C[C_base + x] = (wks){ scaled.real, scaled.imag };
#endif
#else
            C[C_base + x] = (wks){ C_tmp_buffer[y * BLOCK_SIZE + x].real, C_tmp_buffer[y * BLOCK_SIZE + x].imag };
#endif
        }
        C_base += C_row_pitch;
    }

#else

    for (ulong k = 0; k < cols; k += 1) {
        ulong base_index = A_base_index + k * A_row_pitch;
        ulong tile_index = 0;

        for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
                A_tmp_buffer[tile_index + x] = A_packed[base_index + x];
            }
            base_index += BLOCK_SIZE;
            tile_index += BLOCK_SIZE;
        }

        base_index = B_base_index + k * B_row_pitch;
        tile_index = 0;
        for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
            __attribute__((opencl_unroll_hint))
            for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
                B_tmp_buffer[tile_index + x] = B_packed[base_index + x];
            }
            base_index += BLOCK_SIZE;
            tile_index += BLOCK_SIZE;
        }

#if WK_VECTOR_WIDTH <= 8 || BLOCK_SIZE == 2
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

                    C11 = A11 * B11 + A12 * B21 + C11;
                    C12 = A11 * B12 + A12 * B22 + C12;
                    C21 = A21 * B11 + A22 * B21 + C21;
                    C22 = A21 * B12 + A22 * B22 + C22;
                }

                base_index = y * BLOCK_SIZE + x;
                C_tmp_buffer[base_index] += C11;
                C_tmp_buffer[base_index + 1] += C12;
                C_tmp_buffer[base_index + BLOCK_SIZE] += C21;
                C_tmp_buffer[base_index + BLOCK_SIZE + 1] += C22;
            }
        }
#else
        for (ulong y = 0; y < BLOCK_SIZE; y += 4) {
            for (ulong x = 0; x < BLOCK_SIZE; x += 4) {
                wk C11 = (wk)(0);
                wk C12 = (wk)(0);
                wk C13 = (wk)(0);
                wk C14 = (wk)(0);
                wk C21 = (wk)(0);
                wk C22 = (wk)(0);
                wk C23 = (wk)(0);
                wk C24 = (wk)(0);
                wk C31 = (wk)(0);
                wk C32 = (wk)(0);
                wk C33 = (wk)(0);
                wk C34 = (wk)(0);
                wk C41 = (wk)(0);
                wk C42 = (wk)(0);
                wk C43 = (wk)(0);
                wk C44 = (wk)(0);

                __attribute__((opencl_unroll_hint))
                for (ulong z = 0; z < BLOCK_SIZE; z += 4) {
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

                    C11 = A11 * B11 + A12 * B21 + A13 * B31 + A14 * B41 + C11;
                    C12 = A11 * B12 + A12 * B22 + A13 * B32 + A14 * B42 + C12;
                    C13 = A11 * B13 + A12 * B23 + A13 * B33 + A14 * B43 + C13;
                    C14 = A11 * B14 + A12 * B24 + A13 * B34 + A14 * B44 + C14;

                    C21 = A21 * B11 + A22 * B21 + A23 * B31 + A24 * B41 + C21;
                    C22 = A21 * B12 + A22 * B22 + A23 * B32 + A24 * B42 + C22;
                    C23 = A21 * B13 + A22 * B23 + A23 * B33 + A24 * B43 + C23;
                    C24 = A21 * B14 + A22 * B24 + A23 * B34 + A24 * B44 + C24;

                    C31 = A31 * B11 + A32 * B21 + A33 * B31 + A34 * B41 + C31;
                    C32 = A31 * B12 + A32 * B22 + A33 * B32 + A34 * B42 + C32;
                    C33 = A31 * B13 + A32 * B23 + A33 * B33 + A34 * B43 + C33;
                    C34 = A31 * B14 + A32 * B24 + A33 * B34 + A34 * B44 + C34;

                    C41 = A41 * B11 + A42 * B21 + A43 * B31 + A44 * B41 + C41;
                    C42 = A41 * B12 + A42 * B22 + A43 * B32 + A44 * B42 + C42;
                    C43 = A41 * B13 + A42 * B23 + A43 * B33 + A44 * B43 + C43;
                    C44 = A41 * B14 + A42 * B24 + A43 * B34 + A44 * B44 + C44;
                }

                base_index = y * BLOCK_SIZE + x;
                C_tmp_buffer[base_index] += C11;
                C_tmp_buffer[base_index + 1] += C12;
                C_tmp_buffer[base_index + 2] += C13;
                C_tmp_buffer[base_index + 3] += C14;

                base_index += BLOCK_SIZE;
                C_tmp_buffer[base_index] += C21;
                C_tmp_buffer[base_index + 1] += C22;
                C_tmp_buffer[base_index + 2] += C23;
                C_tmp_buffer[base_index + 3] += C24;

                base_index += BLOCK_SIZE;
                C_tmp_buffer[base_index] += C31;
                C_tmp_buffer[base_index + 1] += C32;
                C_tmp_buffer[base_index + 2] += C33;
                C_tmp_buffer[base_index + 3] += C34;

                base_index += BLOCK_SIZE;
                C_tmp_buffer[base_index] += C41;
                C_tmp_buffer[base_index + 1] += C42;
                C_tmp_buffer[base_index + 2] += C43;
                C_tmp_buffer[base_index + 3] += C44;
            }
        }
#endif
    }

    ulong C_base = C_row * C_row_pitch + C_col;
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

#endif
}

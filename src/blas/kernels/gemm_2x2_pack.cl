#include "wekua.h"

__kernel void gemm(
    __global const wk *const restrict A,
    __global const wk *const restrict B,

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

    const ulong C_row = i << 1;
    const ulong C_col = j << 1;

    ulong A_base_index = i * A_slice_pitch;
    ulong B_base_index = j * B_slice_pitch;

#if WK_COMPLEX
    wk C11 = {0, 0};
    wk C12 = {0, 0};
    wk C21 = {0, 0};
    wk C22 = {0, 0};

    COMPLEX_MUL_K(T)
#elif WK_VECTOR_WIDTH == 1
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

    for (ulong k=0; k<cols; k += 2) {
        const wk A11 = A[A_base_index];
        const wk A12 = A[A_base_index + 1];
        const wk A21 = A[A_base_index + 2];
        const wk A22 = A[A_base_index + 3];

        const wk B11 = B[B_base_index];
        const wk B21 = B[B_base_index + 1];
        const wk B12 = B[B_base_index + 2];
        const wk B22 = B[B_base_index + 3];

#if WK_COMPLEX
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
#else
        C11 = A11 * B11 + A12 * B21 + C11;
        C12 = A11 * B12 + A12 * B22 + C12;
        C21 = A21 * B11 + A22 * B21 + C21;
        C22 = A21 * B12 + A22 * B22 + C22;
#endif

        A_base_index += A_row_pitch;
        B_base_index += B_row_pitch;
    }

    const ulong C_index = C_row * C_row_pitch + C_col;
    const ulong C_index2 = C_index + C_row_pitch;

#if WK_COMPLEX
#if HAS_ALPHA
#if HAS_BETA
    wk scaled;
    wk old_val;
    wk beta_scaled;

    COMPLEX_MUL(C11, alpha, scaled);
    old_val = C[C_index];
    COMPLEX_MUL(old_val, beta, beta_scaled);
    C[C_index] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };

    COMPLEX_MUL(C12, alpha, scaled);
    old_val = C[C_index + 1];
    COMPLEX_MUL(old_val, beta, beta_scaled);
    C[C_index + 1] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };

    COMPLEX_MUL(C21, alpha, scaled);
    old_val = C[C_index2];
    COMPLEX_MUL(old_val, beta, beta_scaled);
    C[C_index2] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };

    COMPLEX_MUL(C22, alpha, scaled);
    old_val = C[C_index2 + 1];
    COMPLEX_MUL(old_val, beta, beta_scaled);
    C[C_index2 + 1] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };
#else
    wk scaled;

    COMPLEX_MUL(C11, alpha, scaled);
    C[C_index] = scaled;

    COMPLEX_MUL(C12, alpha, scaled);
    C[C_index + 1] = scaled;

    COMPLEX_MUL(C21, alpha, scaled);
    C[C_index2] = scaled;

    COMPLEX_MUL(C22, alpha, scaled);
    C[C_index2 + 1] = scaled;
#endif
#else
    C[C_index] = C11;
    C[C_index + 1] = C12;
    C[C_index2] = C21;
    C[C_index2 + 1] = C22;
#endif
#elif WK_VECTOR_WIDTH == 1
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

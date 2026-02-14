#include "wekua.h"

// TODO: Add support for big endian devices

// Winograd-Waksman implementation
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

#if A_TRANS == 0
    const ulong row_A = i*A_row_pitch;
    const ulong next_row_A = row_A + A_row_pitch;
#endif

#if B_TRANS
    const ulong row_B = j*B_row_pitch;
    const ulong next_row_B = row_B + B_row_pitch;
#endif

#if WK_COMPLEX

    wk C11 = {0, 0};
    wk C12 = {0, 0};
    wk C21 = {0, 0};
    wk C22 = {0, 0};

    COMPLEX_MUL_K(T)

    for (ulong k=0; k<cols; k += 2) {
#if A_TRANS
        const ulong A_index = k*A_row_pitch + i;
        const ulong A_index2 = A_index + A_row_pitch;

        const wk A11 = A[A_index];
        const wk A21 = A[A_index + 1];
        const wk A12 = A[A_index2];
        const wk A22 = A[A_index2 + 1];
#else
        const wk A11 = A[row_A + k];
        const wk A12 = A[row_A + k + 1];
        const wk A21 = A[next_row_A + k];
        const wk A22 = A[next_row_A + k + 1];
#endif

#if B_TRANS
        const wk B11 = B[row_B + k];
        const wk B21 = B[row_B + k + 1];
        const wk B12 = B[next_row_B + k];
        const wk B22 = B[next_row_B + k + 1];
#else
        const ulong B_index = k*B_row_pitch + j;
        const ulong B_index2 = B_index + B_row_pitch;

        const wk B11 = B[B_index];
        const wk B12 = B[B_index + 1];
        const wk B21 = B[B_index2];
        const wk B22 = B[B_index2 + 1];
#endif

        // Strassen temporaries (component-wise add/sub on .real/.imag)
        wk t0 = { B22.real - B12.real, B22.imag - B12.imag };
        wk t1 = { B11.real + t0.real,  B11.imag + t0.imag  };
        wk t2 = { A11.real - A21.real, A11.imag - A21.imag };
        wk t3 = { t2.real - A22.real,  t2.imag - A22.imag  };

        wk m0, m1, m2, m3, m4, m5, m6;

        COMPLEX_MUL(A11, B11, m0);

        COMPLEX_MUL(A12, B21, m1);

        wk diff1 = { t1.real - B21.real, t1.imag - B21.imag };
        COMPLEX_MUL(A22, diff1, m2);

        COMPLEX_MUL(t2, t0, m3);

        wk sum_a = { A21.real + A22.real, A21.imag + A22.imag };
        wk diff2 = { B12.real - B11.real, B12.imag - B11.imag };
        COMPLEX_MUL(sum_a, diff2, m4);

        wk t3_a12 = { t3.real + A12.real, t3.imag + A12.imag };
        COMPLEX_MUL(t3_a12, B22, m5);

        COMPLEX_MUL(t3, t1, m6);

        // p0 = m0 - m6, p1 = p0 + m3
        wk p0 = { m0.real - m6.real, m0.imag - m6.imag };
        wk p1 = { p0.real + m3.real, p0.imag + m3.imag };

        C11.real += m0.real + m1.real;
        C11.imag += m0.imag + m1.imag;

        C12.real += p0.real + m4.real + m5.real;
        C12.imag += p0.imag + m4.imag + m5.imag;

        C21.real += p1.real - m2.real;
        C21.imag += p1.imag - m2.imag;

        C22.real += p1.real + m4.real;
        C22.imag += p1.imag + m4.imag;
    }

    const ulong C_index = i*C_row_pitch + j;
    const ulong C_index2 = C_index + C_row_pitch;

#if HAS_ALPHA
#if HAS_BETA
    wks scaled;
    wks beta_scaled;
    wks c_old;

    COMPLEX_MUL(C11, alpha, scaled);
    c_old = C[C_index];
    COMPLEX_MUL(c_old, beta, beta_scaled);
    C[C_index] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };

    COMPLEX_MUL(C12, alpha, scaled);
    c_old = C[C_index + 1];
    COMPLEX_MUL(c_old, beta, beta_scaled);
    C[C_index + 1] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };

    COMPLEX_MUL(C21, alpha, scaled);
    c_old = C[C_index2];
    COMPLEX_MUL(c_old, beta, beta_scaled);
    C[C_index2] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };

    COMPLEX_MUL(C22, alpha, scaled);
    c_old = C[C_index2 + 1];
    COMPLEX_MUL(c_old, beta, beta_scaled);
    C[C_index2 + 1] = (wks){ scaled.real + beta_scaled.real, scaled.imag + beta_scaled.imag };
#else
    wk scaled;

    COMPLEX_MUL(C11, alpha, scaled);
    C[C_index] = (wks){ scaled.real, scaled.imag };

    COMPLEX_MUL(C12, alpha, scaled);
    C[C_index + 1] = (wks){ scaled.real, scaled.imag };

    COMPLEX_MUL(C21, alpha, scaled);
    C[C_index2] = (wks){ scaled.real, scaled.imag };

    COMPLEX_MUL(C22, alpha, scaled);
    C[C_index2 + 1] = (wks){ scaled.real, scaled.imag };
#endif
#else
    C[C_index] = (wks){ C11.real, C11.imag };
    C[C_index + 1] = (wks){ C12.real, C12.imag };
    C[C_index2] = (wks){ C21.real, C21.imag };
    C[C_index2 + 1] = (wks){ C22.real, C22.imag };
#endif

#else
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

    for (ulong k=0; k<cols; k += 2) {
#if A_TRANS
        const ulong A_index = k*A_row_pitch + i;
        const ulong A_index2 = A_index + A_row_pitch;

        const wk A11 = A[A_index];
        const wk A21 = A[A_index + 1];
        const wk A12 = A[A_index2];
        const wk A22 = A[A_index2 + 1];
#else
        const wk A11 = A[row_A + k];
        const wk A12 = A[row_A + k + 1];
        const wk A21 = A[next_row_A + k];
        const wk A22 = A[next_row_A + k + 1];
#endif

#if B_TRANS
        const wk B11 = B[row_B + k];
        const wk B21 = B[row_B + k + 1];
        const wk B12 = B[next_row_B + k];
        const wk B22 = B[next_row_B + k + 1];
#else
        const ulong B_index = k*B_row_pitch + j;
        const ulong B_index2 = B_index + B_row_pitch;

        const wk B11 = B[B_index];
        const wk B12 = B[B_index + 1];
        const wk B21 = B[B_index2];
        const wk B22 = B[B_index2 + 1];
#endif

        C11 = A11 * B11 + A12 * B21 + C11;
        C12 = A11 * B12 + A12 * B22 + C12;
        C21 = A21 * B11 + A22 * B21 + C21;
        C22 = A21 * B12 + A22 * B22 + C22;
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

#endif
}

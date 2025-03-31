#include "wekua.h"

// TODO: Add support for big endian devices

// Winograd-Waksman implementation
__kernel void gemm(
    __constant const wk *const restrict A,
    __constant const wk *const restrict B,

#if wk_width == 1
    __global wks *const restrict C,
#else
    __global wk2 *const restrict C,
#endif

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

#if com
#if HAS_ALPHA
    , const wks ialpha
#if HAS_BETA
    , const wks ibeta
#endif
#endif

#endif
) {
    const ulong i = get_global_id(0) << 1;
#if wk_width == 1
    const ulong j = get_global_id(1) << 1;
#else
    const ulong j = get_global_id(1);
#endif

#if A_TRANS == 0
    const ulong row_A = i*A_row_pitch;
    const ulong next_row_A = row_A + A_row_pitch;
#endif

#if B_TRANS
#if wk_width == 1
    const ulong row_B = j*B_row_pitch;
#else
    const ulong row_B = (j << 1)*B_row_pitch;
#endif
    const ulong next_row_B = row_B + B_row_pitch;
#endif

    #if wk_width == 1
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

#if com
    #if wk_width == 1
    wk C11_i = 0;
    wk C12_i = 0;
    wk C21_i = 0;
    wk C22_i = 0;
    #else
    wk C11_i = (wk)(0);
    wk C12_i = (wk)(0);
    wk C21_i = (wk)(0);
    wk C22_i = (wk)(0);
    #endif

    COMPLEX_MUL_K(wk)
    COMPLEX_S_MUL_K(wk)

    for (ulong k=0; k<cols; k += 4) {
#if A_TRANS
        const ulong A_index = (k >> 1)*A_row_pitch + (i << 1);
        const ulong A_index2 = A_index + A_row_pitch;

        const wk A11 = A[A_index];
        const wk A11_i = A[A_index + 1];

        const wk A21 = A[A_index + 2];
        const wk A21_i = A[A_index + 3];

        const wk A12 = A[A_index2];
        const wk A12_i = A[A_index2 + 1];

        const wk A22 = A[A_index2 + 2];
        const wk A22_i = A[A_index2 + 3];
#else
        const wk A11 = A[row_A + k];
        const wk A11_i = A[row_A + k + 1];

        const wk A12 = A[row_A + k + 2];
        const wk A12_i = A[row_A + k + 3];

        const wk A21 = A[next_row_A + k];
        const wk A21_i = A[next_row_A + k + 1];

        const wk A22 = A[next_row_A + k + 2];
        const wk A22_i = A[next_row_A + k + 3];
#endif

#if B_TRANS
        const wk B11 = B[row_B + k];
        const wk B11_i = B[row_B + k + 1];

        const wk B21 = B[row_B + k + 2];
        const wk B21_i = B[row_B + k + 3];

        const wk B12 = B[next_row_B + k];
        const wk B12_i = B[next_row_B + k + 1];

        const wk B22 = B[next_row_B + k + 2];
        const wk B22_i = B[next_row_B + k + 3];
#else
        const ulong B_index = (k >> 1)*B_row_pitch + (j << 1);
        const ulong B_index2 = B_index + B_row_pitch;

        const wk B11 = B[B_index];
        const wk B11_i = B[B_index + 1];

        const wk B12 = B[B_index + 2];
        const wk B12_i = B[B_index + 3];

        const wk B21 = B[B_index2];
        const wk B21_i = B[B_index2 + 1];

        const wk B22 = B[B_index2 + 2];
        const wk B22_i = B[B_index2 + 3];
#endif

        const wk t0 = B22 - B12;
        const wk t0_i = B22_i - B12_i;

        const wk t1 = B11 + t0;
        const wk t1_i = B11_i + t0_i;

        const wk t2 = A11 - A21;
        const wk t2_i = A11_i - A21_i;

        const wk t3 = t2 - A22;
        const wk t3_i = t2_i - A22_i;

        wk m0 = A11, m0_i = A11_i;
        COMPLEX_MUL(m0, m0_i, B11, B11_i);

        wk m1 = A12, m1_i = A12_i;
        COMPLEX_MUL(m1, m1_i, B21, B21_i);

        const wk diff1 = t1 - B21;
        const wk diff1_i = t1_i - B21_i;
        wk m2 = A22, m2_i = A22_i;
        COMPLEX_MUL(m2, m2_i, diff1, diff1_i);

        wk m3 = t2, m3_i = t2_i;
        COMPLEX_MUL(m3, m3_i, t0, t0_i);

        const wk diff2 = B12 - B11;
        const wk diff2_i = B12_i - B11_i;
        wk m4 = A21 + A22, m4_i = A21_i + A22_i;
        COMPLEX_MUL(m4, m4_i, diff2, diff2_i);

        wk m5 = t3 + A12, m5_i = t3_i + A12_i;
        COMPLEX_MUL(m5, m5_i, B22, B22_i);

        wk m6 = t3, m6_i = t3_i;
        COMPLEX_MUL(m6, m6_i, t1, t1_i);

        const wk p0 = m0 - m6;
        const wk p0_i = m0_i - m6_i;

        const wk p1 = p0 + m3;
        const wk p1_i = p0_i + m3_i;

        C11 += m0 + m1;
        C11_i += m0_i + m1_i;

        C12 += p0 + m4 + m5;
        C12_i += p0_i + m4_i + m5_i;

        C21 += p1 - m2;
        C21_i += p1_i - m2_i;

        C22 += p1 + m4;
        C22_i += p1_i + m4_i;
    }

    const ulong C_index = i*C_row_pitch + (j << 1);
    const ulong C_index2 = C_index + C_row_pitch;

#if wk_width == 1

#if HAS_ALPHA
#if HAS_BETA
    wks br_value = beta;
    wks bi_value = ibeta;

    wks cr_value = C[C_index];
    wks ci_value = C[C_index + 1];

    COMPLEX_S_MUL(br_value, bi_value, cr_value, ci_value);
    COMPLEX_MUL(C11, C11_i, alpha, ialpha);

    C[C_index] = C11 + br_value;
    C[C_index + 1] = C11_i + bi_value;

    br_value = beta;
    bi_value = ibeta;

    cr_value = C[C_index + 2];
    ci_value = C[C_index + 3];

    COMPLEX_S_MUL(br_value, bi_value, cr_value, ci_value);
    COMPLEX_MUL(C12, C12_i, alpha, ialpha);

    C[C_index + 2] = C12 + br_value;
    C[C_index + 3] = C12_i + bi_value;

    br_value = beta;
    bi_value = ibeta;

    cr_value = C[C_index2];
    ci_value = C[C_index2 + 1];

    COMPLEX_S_MUL(br_value, bi_value, cr_value, ci_value);
    COMPLEX_MUL(C21, C21_i, alpha, ialpha);

    C[C_index2] = C21 + br_value;
    C[C_index2 + 1] = C21_i + bi_value;

    br_value = beta;
    bi_value = ibeta;

    cr_value = C[C_index2 + 2];
    ci_value = C[C_index2 + 3];

    COMPLEX_S_MUL(br_value, bi_value, cr_value, ci_value);
    COMPLEX_MUL(C22, C22_i, alpha, ialpha);

    C[C_index2 + 2] = C22 + br_value;
    C[C_index2 + 3] = C22_i + bi_value;
#else
    COMPLEX_MUL(C11, C11_i, alpha, ialpha);
    C[C_index] = C11;
    C[C_index + 1] = C11_i;

    COMPLEX_MUL(C12, C12_i, alpha, ialpha);
    C[C_index + 2] = C12;
    C[C_index + 3] = C12_i;

    COMPLEX_MUL(C21, C21_i, alpha, ialpha);
    C[C_index2] = C21;
    C[C_index2 + 1] = C21_i;

    COMPLEX_MUL(C22, C22_i, alpha, ialpha);
    C[C_index2 + 2] = C22;
    C[C_index2 + 3] = C22_i;
#endif
#else
    C[C_index] = C11;
    C[C_index + 1] = C11_i;

    C[C_index + 2] = C12;
    C[C_index + 3] = C12_i;

    C[C_index2] = C21;
    C[C_index2 + 1] = C21_i;

    C[C_index2 + 2] = C22;
    C[C_index2 + 3] = C22_i;
#endif

#else

#if HAS_ALPHA
#if HAS_BETA
    wks br_value = beta;
    wks bi_value = ibeta;

    wks cr_value = C[C_index];
    wks ci_value = C[C_index + 1];

    COMPLEX_S_MUL(br_value, bi_value, cr_value, ci_value);
    COMPLEX_MUL(C11, C11_i, alpha, ialpha);

    C[C_index] = sum(C11) + br_value;
    C[C_index + 1] = sum(C11_i) + bi_value;

    br_value = beta;
    bi_value = ibeta;

    cr_value = C[C_index + 2];
    ci_value = C[C_index + 3];

    COMPLEX_S_MUL(br_value, bi_value, cr_value, ci_value);
    COMPLEX_MUL(C12, C12_i, alpha, ialpha);

    C[C_index + 2] = sum(C12) + br_value;
    C[C_index + 3] = sum(C12_i) + bi_value;

    br_value = beta;
    bi_value = ibeta;

    cr_value = C[C_index2];
    ci_value = C[C_index2 + 1];

    COMPLEX_S_MUL(br_value, bi_value, cr_value, ci_value);
    COMPLEX_MUL(C21, C21_i, alpha, ialpha);

    C[C_index2] = sum(C21) + br_value;
    C[C_index2 + 1] = sum(C21_i) + bi_value;

    br_value = beta;
    bi_value = ibeta;

    cr_value = C[C_index2 + 2];
    ci_value = C[C_index2 + 3];

    COMPLEX_S_MUL(br_value, bi_value, cr_value, ci_value);
    COMPLEX_MUL(C22, C22_i, alpha, ialpha);

    C[C_index2 + 2] = sum(C22) + br_value;
    C[C_index2 + 3] = sum(C22_i) + bi_value;
#else
    COMPLEX_MUL(C11, C11_i, alpha, ialpha);
    C[C_index] = sum(C11);
    C[C_index + 1] = sum(C11_i);

    COMPLEX_MUL(C12, C12_i, alpha, ialpha);
    C[C_index + 2] = sum(C12);
    C[C_index + 3] = sum(C12_i);

    COMPLEX_MUL(C21, C21_i, alpha, ialpha);
    C[C_index2] = sum(C21);
    C[C_index2 + 1] = sum(C21_i);

    COMPLEX_MUL(C22, C22_i, alpha, ialpha);
    C[C_index2 + 2] = sum(C22);
    C[C_index2 + 3] = sum(C22_i);
#endif
#else
    C[C_index] = sum(C11);
    C[C_index + 1] = sum(C11_i);

    C[C_index + 2] = sum(C12);
    C[C_index + 3] = sum(C12_i);

    C[C_index2] = sum(C21);
    C[C_index2 + 1] = sum(C21_i);

    C[C_index2 + 2] = sum(C22);
    C[C_index2 + 3] = sum(C22_i);
#endif
#endif

#else
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

    const ulong C_index = i*C_row_pitch + j;
    const ulong C_index2 = C_index + C_row_pitch;

#if wk_width == 1

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
    C[C_index] = alpha * (wk2)(sum(C11), sum(C12)) + C[C_index] * beta;
    C[C_index2] = alpha * (wk2)(sum(C21), sum(C22)) + C[C_index2] * beta;
    /*
    C[C_index] = alpha*sum(C11) + beta*C[C_index];
    C[C_index + 1] = alpha*sum(C12) + beta*C[C_index + 1];
    C[C_index2] = alpha*sum(C21) + beta*C[C_index2];
    C[C_index2 + 1] = alpha*sum(C22) + beta*C[C_index2 + 1];
    */
#else
    C[C_index] = alpha * (wk2)(sum(C11), sum(C12));
    C[C_index2] = alpha * (wk2)(sum(C21), sum(C22));

    /*
    C[C_index] = alpha*sum(C11);
    C[C_index + 1] = alpha*sum(C12);
    C[C_index2] = alpha*sum(C21);
    C[C_index2 + 1] = alpha*sum(C22);
    */
#endif
#else
    C[C_index] = (wk2)(sum(C11), sum(C12));
    C[C_index2] = (wk2)(sum(C21), sum(C22));

    /*
    C[C_index] = sum(C11);
    C[C_index + 1] = sum(C12);
    C[C_index2] = sum(C21);
    C[C_index2 + 1] = sum(C22);
    */
#endif

#endif

#endif
}

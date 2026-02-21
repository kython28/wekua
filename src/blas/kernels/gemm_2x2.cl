/**
 * =============================================================================
 * GEMM 2x2 — CPU GEMM with fixed 2x2 micro-kernel, unpacked inputs
 * =============================================================================
 *
 * Minimal GEMM kernel for CPU devices. Each work-item computes a 2x2 block
 * of the output matrix C using 4 register accumulators (C11, C12, C21, C22).
 * Reads directly from global memory without local memory or intermediate
 * buffers. Supports optional transpose on A and/or B via compile-time flags.
 *
 * Computes: C = alpha * op(A) * op(B) + beta * C
 * where op(X) = X or X^T depending on A_TRANS / B_TRANS flags.
 *
 * COMPILE-TIME PARAMETERS
 * -----------------------
 * A_TRANS          - 0: A is row-major, 1: A is transposed
 * B_TRANS          - 0: B is column-major access, 1: B is row-major
 * HAS_ALPHA        - 0: alpha=1 (omitted), 1: alpha scaling is applied
 * HAS_BETA         - 0: no beta term, 1: beta * C_old is added (requires HAS_ALPHA)
 * WK_COMPLEX       - 0: scalar/vector types, 1: complex arithmetic
 * WK_VECTOR_WIDTH  - SIMD vector width (1, 2, 4, 8, 16)
 *
 * KERNEL PARAMETERS
 * -----------------
 * A                - Input matrix A (__global, read-only)
 * B                - Input matrix B (__global, read-only)
 * C                - Output matrix C (__global, read-write, scalar type wks)
 * A_row_pitch      - Elements per row in A
 * B_row_pitch      - Elements per row in B
 * C_row_pitch      - Elements per row in C
 * cols             - Shared dimension K (number of columns of op(A) / rows of op(B))
 * alpha            - Scalar multiplier (conditional on HAS_ALPHA)
 * beta             - Scalar multiplier for existing C (conditional on HAS_BETA)
 *
 * NDRANGE (2D)
 * ------------
 * dim 0 (i)  - Output row index / 2  (each WI covers 2 rows)
 * dim 1 (j)  - Output column index / 2  (each WI covers 2 columns)
 * Global IDs are shifted << 1 to get the actual row/column.
 *
 * ALGORITHM
 * ---------
 * 1. Map work-item to 2x2 output block at (i, j) via global_id << 1
 * 2. Pre-compute row offsets for non-transposed matrix access
 * 3. Loop k in steps of 2: load a 2x2 sub-block from each of A and B
 *    - A_TRANS=0: rows are contiguous, columns advance with k
 *    - A_TRANS=1: columns are contiguous, rows advance with k
 *    - B_TRANS=0: columns advance with k (column-major access pattern)
 *    - B_TRANS=1: rows are contiguous, columns advance with k
 * 4. Accumulate 2x2 matrix product into register accumulators
 * 5. Write back to C with optional alpha/beta scaling
 *
 * MEMORY ACCESS PATTERN
 * ---------------------
 * All reads from global memory. Row offsets for A (non-transposed) and B
 * (transposed) are pre-computed outside the loop to avoid redundant
 * multiplications. For vectorized types (WK_VECTOR_WIDTH > 1), the final
 * result uses sum() to reduce vectors to scalars before writing to C.
 *
 * =============================================================================
 */

#include "wekua.h"

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
    // Each work-item computes a 2x2 block; shift IDs to get top-left corner
    const ulong i = get_global_id(0) << 1;
    const ulong j = get_global_id(1) << 1;

    // Pre-compute row base offsets outside the loop to avoid repeated multiplies.
    // A non-transposed: rows i and i+1 are at fixed offsets.
    // B transposed: rows j and j+1 of B^T correspond to columns j and j+1 of B.
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

    // Loop k in steps of 2 to feed the 2x2 micro-kernel with pairs of k-elements
    for (ulong k=0; k<cols; k += 2) {
        // A_TRANS=1: A is stored transposed, so row k of A^T is column k of A.
        // Two consecutive k-values live in adjacent rows of the stored matrix.
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

        // B_TRANS=1: B is stored transposed, read row j of B^T (= column j of B).
        // B_TRANS=0: B is row-major; row k of B contains column elements for C.
        // Note: B11/B12/B21/B22 naming matches the 2x2 sub-block of op(B),
        // where B_ij means row i, col j of the effective (possibly transposed) B.
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
    }

    const ulong C_index = i*C_row_pitch + j;
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

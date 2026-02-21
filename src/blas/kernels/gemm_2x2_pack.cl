/**
 * =============================================================================
 * GEMM 2x2 PACK — CPU GEMM with fixed 2x2 micro-kernel, packed inputs
 * =============================================================================
 *
 * Same 2x2 micro-kernel as gemm_2x2.cl but reads from pre-packed tile buffers
 * instead of raw row-major matrices. Because the packing step has already
 * handled any transpose, there is no A_TRANS/B_TRANS branching.
 *
 * Computes: C = alpha * A_packed * B_packed + beta * C
 *
 * COMPILE-TIME PARAMETERS
 * -----------------------
 * HAS_ALPHA        - 0: alpha=1 (omitted), 1: alpha scaling is applied
 * HAS_BETA         - 0: no beta term, 1: beta * C_old is added (requires HAS_ALPHA)
 * WK_COMPLEX       - 0: scalar/vector types, 1: complex arithmetic
 * WK_VECTOR_WIDTH  - SIMD vector width (1, 2, 4, 8, 16)
 *
 * KERNEL PARAMETERS
 * -----------------
 * A                - Packed matrix A (__global, read-only)
 * B                - Packed matrix B (__global, read-only)
 * C                - Output matrix C in row-major layout (__global, read-write)
 * A_slice_pitch    - Stride between tile-groups of A (one per output tile-row)
 * A_row_pitch      - Stride between consecutive k-tiles within an A tile-group
 * B_slice_pitch    - Stride between tile-groups of B (one per output tile-col)
 * B_row_pitch      - Stride between consecutive k-tiles within a B tile-group
 * C_row_pitch      - Elements per row in the output matrix C
 * cols             - Number of k-tiles to iterate over
 * alpha            - Scalar multiplier (conditional on HAS_ALPHA)
 * beta             - Scalar multiplier for existing C (conditional on HAS_BETA)
 *
 * NDRANGE (2D)
 * ------------
 * dim 0 (i)  - Tile-row index (maps to 2 output rows via C_row = i << 1)
 * dim 1 (j)  - Tile-col index (maps to 2 output cols via C_col = j << 1)
 *
 * ALGORITHM
 * ---------
 * 1. Map work-item to packed tile-groups: A_base = i * A_slice_pitch,
 *    B_base = j * B_slice_pitch
 * 2. Loop k from 0 to cols in steps of 1 (each k is one pre-packed 2x2 tile):
 *    a. Load 4 elements from A_packed (A11, A12, A21, A22) sequentially
 *    b. Load 4 elements from B_packed (B11, B21, B12, B22)
 *    c. Accumulate 2x2 product into C11..C22 registers
 *    d. Advance base indices by row_pitch to the next k-tile
 * 3. Write back to C at row-major position with optional alpha/beta scaling
 *
 * MEMORY ACCESS PATTERN
 * ---------------------
 * Packed buffers are laid out so that each 2x2 tile is 4 contiguous elements.
 * The loop advances by row_pitch per k-step, which is the stride between
 * consecutive k-tiles within a slice. No transpose branching is needed because
 * the packing kernel already arranged the data in the correct orientation.
 *
 * =============================================================================
 */

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

    // slice_pitch selects the tile-group for this work-item's output tile-row/col.
    // Each tile-group contains all k-tiles for one row (A) or column (B) of tiles.
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

    // Loop over k-tiles; each k-step is one packed 2x2 tile (4 contiguous elements)
    for (ulong k=0; k<cols; k += 1) {
        // Packed layout: [A11, A12, A21, A22] — row-major within the 2x2 tile
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

        // Advance to the next k-tile within the tile-group
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

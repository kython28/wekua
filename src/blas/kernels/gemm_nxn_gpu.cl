/**
 * =============================================================================
 * GEMM NxN GPU — GPU GEMM with local memory tiling, unpacked inputs
 * =============================================================================
 *
 * GPU-optimized GEMM kernel using __local memory for shared tile storage and
 * 2x2 register tiling. Each work-item computes a 2x2 sub-block of C, so the
 * workgroup collectively computes a BLOCK_SIZE x BLOCK_SIZE output tile.
 *
 * The 2x2 register tiling means:
 *   - Global/local IDs are shifted << 1 (each WI covers 2 rows and 2 cols)
 *   - Workgroup size is (BLOCK_SIZE/2) x (BLOCK_SIZE/2)
 *   - Each WI loads 4 elements into local memory (collaborative loading)
 *   - Inner loop steps by 2 (kk += 2) to feed the 2x2 micro-kernel
 *   - Compute-to-memory ratio: O(2 * BLOCK_SIZE) MADs per load
 *
 * Computes: C = alpha * op(A) * op(B) + beta * C
 *
 * COMPILE-TIME PARAMETERS
 * -----------------------
 * BLOCK_SIZE        - Tile dimension (2, 4, 8, 16, 32, 64)
 * A_TRANS           - 0: A is row-major, 1: A is transposed
 * B_TRANS           - 0: B is column-major access, 1: B is row-major
 * HAS_ALPHA         - 0: alpha=1 (omitted), 1: alpha scaling is applied
 * HAS_BETA          - 0: no beta term, 1: beta * C_old is added
 * WK_COMPLEX        - 0: scalar/vector types, 1: complex arithmetic
 * WK_VECTOR_WIDTH   - SIMD vector width (1, 2, 4, 8, 16)
 * WK_CACHE_LINE_SIZE - Alignment for local tile buffers
 *
 * KERNEL PARAMETERS
 * -----------------
 * A                - Input matrix A (__global, read-only)
 * B                - Input matrix B (__global, read-only)
 * C                - Output matrix C (__global, read-write, scalar type wks)
 * A_row_pitch      - Elements per row in A
 * B_row_pitch      - Elements per row in B
 * C_row_pitch      - Elements per row in C
 * cols             - Shared dimension K (padded to multiple of BLOCK_SIZE)
 * alpha            - Scalar multiplier (conditional on HAS_ALPHA)
 * beta             - Scalar multiplier for existing C (conditional on HAS_BETA)
 *
 * NDRANGE (2D)
 * ------------
 * dim 0  - Row tile index / 2  (global_size = M / 2, local_size = BLOCK_SIZE / 2)
 * dim 1  - Col tile index / 2  (global_size = N / 2, local_size = BLOCK_SIZE / 2)
 * Each work-item maps to a 2x2 output block at (global_id(0)<<1, global_id(1)<<1).
 *
 * ALGORITHM
 * ---------
 * 1. Each WI maps to output coordinates (i, j) = (gid0 << 1, gid1 << 1)
 * 2. Compute local tile indices:
 *    - A_local_tile_index = li * BLOCK_SIZE + lj  (row-major storage)
 *    - B_local_tile_index = lj * BLOCK_SIZE + li  (transposed storage for
 *      contiguous column access in the micro-kernel)
 * 3. Loop k from 0 to cols in steps of BLOCK_SIZE:
 *    a. Collaborative load: each WI loads a 2x2 block into A_local and B_local
 *       - A_TRANS/B_TRANS determines the global memory access pattern
 *       - B is stored transposed in local memory so micro-kernel reads columns
 *         of B as contiguous rows
 *    b. barrier(CLK_LOCAL_MEM_FENCE)
 *    c. Inner loop kk from 0 to BLOCK_SIZE in steps of 2:
 *       - Load 2x2 from A_local (rows li, li+1; cols kk, kk+1)
 *       - Load 2x2 from B_local (rows lj, lj+1; cols kk, kk+1)
 *       - 2x2 multiply-accumulate into C11, C12, C21, C22
 *    d. barrier(CLK_LOCAL_MEM_FENCE)
 * 4. Write 2x2 result to C with optional alpha/beta scaling
 *
 * MEMORY ACCESS PATTERN
 * ---------------------
 * Local memory: two BLOCK_SIZE^2 tiles for A and B, shared by the workgroup.
 * B is stored transposed in local memory (B_local_tile_index = lj*BS + li)
 * so that the micro-kernel's inner loop reads B columns as contiguous rows,
 * avoiding strided local memory access.
 * Two barriers per k-step: one after loading tiles, one after computing
 * (to prevent the next load from overwriting data still in use).
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
    // 2x2 register tiling: each WI covers 2 rows and 2 columns
    const ulong i = get_global_id(0) << 1;
    const ulong j = get_global_id(1) << 1;

    const ulong li = get_local_id(0) << 1;
    const ulong lj = get_local_id(1) << 1;

    local wk A_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    local wk B_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));

    // A: row-major in local memory — row li at offset li * BLOCK_SIZE
    const ulong A_local_tile_index = li * BLOCK_SIZE + lj;
    // B: stored transposed in local memory (lj * BLOCK_SIZE + li instead of
    // li * BLOCK_SIZE + lj) so that micro-kernel reads B columns as
    // contiguous rows, avoiding strided local memory access
    const ulong B_local_tile_index = lj * BLOCK_SIZE + li;

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


    // Pre-compute row bases in local memory for the micro-kernel
    const ulong A_base_index = li * BLOCK_SIZE;
    const ulong B_base_index = lj * BLOCK_SIZE;
    for (ulong k = 0; k < cols; k += BLOCK_SIZE) {
        // --- Collaborative tile loading ---
        // Each WI loads a 2x2 block into local memory (total BLOCK_SIZE^2 elements).
        ulong base_index;

        // A loading: A_TRANS determines whether we read rows or columns.
        // A_TRANS=1: column k of A becomes row k of A^T. Two k-values are in
        //            adjacent rows of stored A, so we step by A_row_pitch.
        // A_TRANS=0: row i of A, columns k+lj and k+lj+1 are adjacent.
#if A_TRANS
        base_index = (k + lj) * A_row_pitch + i;
        A_tmp_buffer[A_local_tile_index] = A[base_index];
        A_tmp_buffer[A_local_tile_index + 1] = A[base_index + A_row_pitch];
        A_tmp_buffer[A_local_tile_index + BLOCK_SIZE] = A[base_index + 1];
        A_tmp_buffer[A_local_tile_index + BLOCK_SIZE + 1] = A[base_index + A_row_pitch + 1];
#else
        base_index = i * A_row_pitch + k + lj;
        A_tmp_buffer[A_local_tile_index] = A[base_index];
        A_tmp_buffer[A_local_tile_index + 1] = A[base_index + 1];
        A_tmp_buffer[A_local_tile_index + BLOCK_SIZE] = A[base_index + A_row_pitch];
        A_tmp_buffer[A_local_tile_index + BLOCK_SIZE + 1] = A[base_index + A_row_pitch + 1];
#endif

        // B loading: stored transposed in local memory (B_local_tile_index swaps row/col).
        // B_TRANS=1: B^T is row-major, so row j of B^T is read normally.
        // B_TRANS=0: B is row-major; we write transposed so micro-kernel reads
        //            B columns as contiguous rows.
#if B_TRANS
        base_index = j * B_row_pitch + k + li;
        B_tmp_buffer[B_local_tile_index] = B[base_index];
        B_tmp_buffer[B_local_tile_index + 1] = B[base_index + 1];
        B_tmp_buffer[B_local_tile_index + BLOCK_SIZE] = B[base_index + B_row_pitch];
        B_tmp_buffer[B_local_tile_index + BLOCK_SIZE + 1] = B[base_index + B_row_pitch + 1];
#else
        base_index = (k + li) * B_row_pitch + j;
        B_tmp_buffer[B_local_tile_index] = B[base_index];
        B_tmp_buffer[B_local_tile_index + 1] = B[base_index + B_row_pitch];
        B_tmp_buffer[B_local_tile_index + BLOCK_SIZE] = B[base_index + 1];
        B_tmp_buffer[B_local_tile_index + BLOCK_SIZE + 1] = B[base_index + B_row_pitch + 1];
#endif
        // Ensure all WIs have finished loading before computing
        barrier(CLK_LOCAL_MEM_FENCE);

        // Inner loop: step by 2 to feed the 2x2 micro-kernel with pairs of k-elements
        __attribute__((opencl_unroll_hint))
        for (ulong kk = 0; kk < BLOCK_SIZE; kk += 2) {
            const wk A11 = A_tmp_buffer[A_base_index + kk];
            const wk A12 = A_tmp_buffer[A_base_index + kk + 1];
            const wk A21 = A_tmp_buffer[A_base_index + BLOCK_SIZE + kk];
            const wk A22 = A_tmp_buffer[A_base_index + BLOCK_SIZE + kk + 1];

            const wk B11 = B_tmp_buffer[B_base_index + kk];
            const wk B21 = B_tmp_buffer[B_base_index + kk + 1];
            const wk B12 = B_tmp_buffer[B_base_index + BLOCK_SIZE + kk];
            const wk B22 = B_tmp_buffer[B_base_index + BLOCK_SIZE + kk + 1];

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

        // Ensure all WIs are done computing before next iteration overwrites local tiles
        barrier(CLK_LOCAL_MEM_FENCE);
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

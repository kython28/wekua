/**
 * =============================================================================
 * GEMM NxN PACK GPU — GPU GEMM with local memory, packed inputs
 * =============================================================================
 *
 * Combines pre-packed tile layout with GPU local memory tiling and 2x2 register
 * tiling. No A_TRANS/B_TRANS branching — packing already handled transpose.
 * Each workgroup processes one output tile of BLOCK_SIZE x BLOCK_SIZE, with
 * each work-item computing a 2x2 sub-block.
 *
 * Computes: C = alpha * A_packed * B_packed + beta * C
 *
 * COMPILE-TIME PARAMETERS
 * -----------------------
 * BLOCK_SIZE        - Tile dimension (2, 4, 8, 16, 32, 64)
 * HAS_ALPHA         - 0: alpha=1 (omitted), 1: alpha scaling is applied
 * HAS_BETA          - 0: no beta term, 1: beta * C_old is added
 * WK_COMPLEX        - 0: scalar/vector types, 1: complex arithmetic
 * WK_VECTOR_WIDTH   - SIMD vector width (1, 2, 4, 8, 16)
 * WK_CACHE_LINE_SIZE - Alignment for local tile buffers
 *
 * KERNEL PARAMETERS
 * -----------------
 * A_packed         - Packed matrix A (__global, read-only)
 * B_packed         - Packed matrix B (__global, read-only)
 * C                - Output matrix C in row-major layout (__global, read-write)
 * A_slice_pitch    - Stride between tile-groups of A (one per output tile-row)
 * A_row_pitch      - Stride between consecutive k-tiles within an A tile-group
 * B_slice_pitch    - Stride between tile-groups of B (one per output tile-col)
 * B_row_pitch      - Stride between consecutive k-tiles within a B tile-group
 * C_row_pitch      - Elements per row in output C
 * cols             - Number of k-tiles to iterate over
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
 * 1. Compute packed_depth from global output coordinates:
 *    A_packed_depth = tile-row of this workgroup, B_packed_depth = tile-col
 * 2. Compute base addresses into packed buffers using slice_pitch and
 *    local tile indices
 * 3. Loop k from 0 to cols:
 *    a. Each WI loads 4 elements from packed A and B into local memory
 *       - A: row-major in local memory (A_local_tile_index = li*BS + lj)
 *       - B: transposed in local memory (B_local_tile_index = lj*BS + li)
 *    b. barrier(CLK_LOCAL_MEM_FENCE)
 *    c. Inner loop kk += 2: 2x2 micro-kernel over local tiles
 *    d. Advance base indices by row_pitch to next k-tile
 *    e. barrier(CLK_LOCAL_MEM_FENCE)
 * 4. Write 2x2 result to C with optional alpha/beta scaling
 *
 * MEMORY ACCESS PATTERN
 * ---------------------
 * Packed tiles are stored contiguously in memory. Each WI reads 4 elements
 * at packed_base + local offsets, which are spatially close for good
 * coalescing. B is stored transposed in local memory for contiguous column
 * access by the micro-kernel. Two barriers per k-step separate load and
 * compute phases.
 *
 * =============================================================================
 */

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
    // 2x2 register tiling: each WI covers 2 rows and 2 columns
    const ulong i = get_global_id(0) << 1;
    const ulong j = get_global_id(1) << 1;

    const ulong li = get_local_id(0) << 1;
    const ulong lj = get_local_id(1) << 1;

    // packed_depth = which tile-group this workgroup belongs to.
    // Equivalent to floor(i / BLOCK_SIZE) — identifies the tile-row (A) or tile-col (B)
    // in the packed layout. slice_pitch * packed_depth jumps to that tile-group.
    const ulong A_packed_depth = (i - i % BLOCK_SIZE) / BLOCK_SIZE;
    const ulong B_packed_depth = (j - j % BLOCK_SIZE) / BLOCK_SIZE;

    local wk A_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    local wk B_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));

    // A: row-major in local memory
    const ulong A_local_tile_index = li * BLOCK_SIZE + lj;
    // B: transposed in local memory for contiguous column access in micro-kernel
    const ulong B_local_tile_index = lj * BLOCK_SIZE + li;

    // Base index into packed buffer: slice selects tile-group, local index selects
    // this WI's 2x2 loading position within the tile
    ulong A_base_index = A_packed_depth * A_slice_pitch + A_local_tile_index;
    ulong B_base_index = B_packed_depth * B_slice_pitch + B_local_tile_index;

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
    const ulong A_tile_base_index = li * BLOCK_SIZE;
    const ulong B_tile_base_index = lj * BLOCK_SIZE;
    for (ulong k = 0; k < cols; k += 1) {
        // --- Collaborative tile loading from packed buffers ---
        // Each WI loads its 2x2 block at A_local_tile_index / B_local_tile_index
        A_tmp_buffer[A_local_tile_index] = A_packed[A_base_index];
        A_tmp_buffer[A_local_tile_index + 1] = A_packed[A_base_index + 1];
        A_tmp_buffer[A_local_tile_index + BLOCK_SIZE] = A_packed[A_base_index + BLOCK_SIZE];
        A_tmp_buffer[A_local_tile_index + BLOCK_SIZE + 1] = A_packed[A_base_index + BLOCK_SIZE + 1];

        B_tmp_buffer[B_local_tile_index] = B_packed[B_base_index];
        B_tmp_buffer[B_local_tile_index + 1] = B_packed[B_base_index + 1];
        B_tmp_buffer[B_local_tile_index + BLOCK_SIZE] = B_packed[B_base_index + BLOCK_SIZE];
        B_tmp_buffer[B_local_tile_index + BLOCK_SIZE + 1] = B_packed[B_base_index + BLOCK_SIZE + 1];
        // Ensure all WIs have finished loading before computing
        barrier(CLK_LOCAL_MEM_FENCE);

        // Inner loop: step by 2 to feed the 2x2 micro-kernel with pairs of k-elements
        __attribute__((opencl_unroll_hint))
        for (ulong kk = 0; kk < BLOCK_SIZE; kk += 2) {
            const wk A11 = A_tmp_buffer[A_tile_base_index + kk];
            const wk A12 = A_tmp_buffer[A_tile_base_index + kk + 1];
            const wk A21 = A_tmp_buffer[A_tile_base_index + BLOCK_SIZE + kk];
            const wk A22 = A_tmp_buffer[A_tile_base_index + BLOCK_SIZE + kk + 1];

            const wk B11 = B_tmp_buffer[B_tile_base_index + kk];
            const wk B21 = B_tmp_buffer[B_tile_base_index + kk + 1];
            const wk B12 = B_tmp_buffer[B_tile_base_index + BLOCK_SIZE + kk];
            const wk B22 = B_tmp_buffer[B_tile_base_index + BLOCK_SIZE + kk + 1];

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

        // Advance to the next k-tile within the packed tile-group
        A_base_index += A_row_pitch;
        B_base_index += B_row_pitch;
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

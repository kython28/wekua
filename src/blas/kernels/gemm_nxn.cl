/**
 * =============================================================================
 * GEMM NxN — CPU GEMM with configurable block size, unpacked inputs
 * =============================================================================
 *
 * General-purpose CPU GEMM kernel supporting any BLOCK_SIZE (4, 8, 16, 32, 64).
 * Each work-item computes one BLOCK_SIZE x BLOCK_SIZE tile of C using private
 * memory buffers for A, B, and C tiles. Two micro-kernel variants are selected
 * at compile time based on vector width:
 *   - 2x2 micro-kernel: used when WK_VECTOR_WIDTH <= 8 or WK_COMPLEX
 *   - 4x4 micro-kernel: used when WK_VECTOR_WIDTH > 8 (e.g., 16-wide vectors)
 *     to better utilize wide SIMD registers
 *
 * Computes: C = alpha * op(A) * op(B) + beta * C
 *
 * COMPILE-TIME PARAMETERS
 * -----------------------
 * BLOCK_SIZE        - Tile dimension (4, 8, 16, 32, 64)
 * STRIDE            - log2(BLOCK_SIZE), used for bit-shift addressing
 * A_TRANS           - 0: A is row-major, 1: A is transposed
 * B_TRANS           - 0: B is column-major access, 1: B is row-major
 * HAS_ALPHA         - 0: alpha=1 (omitted), 1: alpha scaling is applied
 * HAS_BETA          - 0: no beta term, 1: beta * C_old is added
 * WK_COMPLEX        - 0: scalar/vector types, 1: complex arithmetic
 * WK_VECTOR_WIDTH   - SIMD vector width (1, 2, 4, 8, 16)
 * WK_CACHE_LINE_SIZE - Alignment for private tile buffers
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
 * dim 0 (i)  - Output tile-row index  (actual row = global_id(0) << STRIDE)
 * dim 1 (j)  - Output tile-col index  (actual col = global_id(1) << STRIDE)
 *
 * ALGORITHM
 * ---------
 * 1. Map work-item to output tile at (i, j) via global_id << STRIDE
 * 2. Allocate private A_tmp, B_tmp, C_tmp buffers (BLOCK_SIZE^2 each)
 * 3. Loop k from 0 to cols in steps of BLOCK_SIZE:
 *    a. Load A tile using FILL_TILE (normal) or FILL_TRANSPOSED_TILE (transposed)
 *    b. Load B tile — B is always stored transposed in B_tmp_buffer so that the
 *       micro-kernel reads B columns as contiguous rows (better cache behavior)
 *    c. Run nested micro-kernel loops over the tile:
 *       - 2x2 path: y+=2, x+=2, z+=2 inner loop
 *       - 4x4 path: y+=4, x+=4, z+=4 inner loop (16 accumulators)
 *    d. Accumulate partial results into C_tmp_buffer
 * 4. Write C_tmp_buffer to global C with optional alpha/beta scaling
 *
 * MEMORY ACCESS PATTERN
 * ---------------------
 * Private memory (registers/stack): A_tmp, B_tmp, C_tmp buffers.
 * Global memory: sequential tile loads via FILL_TILE macros, final write-back.
 * No local memory is used — this kernel targets CPU devices where private
 * memory maps to registers/L1 cache efficiently.
 *
 * =============================================================================
 */

#include "wekua.h"

/**
 * FILL_TILE — Load a BLOCK_SIZE x BLOCK_SIZE tile from global memory (row-major)
 *
 * Copies a contiguous block of the source matrix into a private tile buffer.
 * Used for A when A_TRANS=0, and for B when B_TRANS=1 (B^T is row-major in
 * the output column direction).
 */
#define FILL_TILE(tile, values, row_index, col_index, row_pitch) \
    base_index = row_index * row_pitch + col_index; \
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) { \
        __attribute__((opencl_unroll_hint)) \
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) { \
            tile[y * BLOCK_SIZE + x] = values[base_index + x]; \
        } \
        base_index += row_pitch; \
    }

/**
 * FILL_TRANSPOSED_TILE — Load a tile while transposing it
 *
 * Reads from the source matrix with transposed access (stepping by row_pitch
 * in the inner loop) and writes into the tile in row-major order. Used for:
 * - A when A_TRANS=1: reads column-wise from the stored A
 * - B when B_TRANS=0: transposes B so the micro-kernel can read B columns
 *   as contiguous rows, which improves spatial locality
 */
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
        // A_TRANS=1: A is stored transposed, so we read it transposed to get the
        // correct orientation. A_TRANS=0: normal row-major read.
#if A_TRANS
        FILL_TRANSPOSED_TILE(A_tmp_buffer, A, k, i, A_row_pitch)
#else
        FILL_TILE(A_tmp_buffer, A, i, k, A_row_pitch)
#endif

        // B is ALWAYS stored transposed in B_tmp_buffer regardless of B_TRANS.
        // This ensures the micro-kernel reads B columns as contiguous memory.
        // B_TRANS=1: B is already transposed in memory, use normal FILL_TILE.
        // B_TRANS=0: B is row-major, transpose during load.
#if B_TRANS
        FILL_TILE(B_tmp_buffer, B, j, k, B_row_pitch)
#else
        FILL_TRANSPOSED_TILE(B_tmp_buffer, B, k, j, B_row_pitch)
#endif

        // Select micro-kernel size based on vector width:
        // - 2x2: for narrow vectors (<=8) or complex types; 4 accumulators
        // - 4x4: for wide vectors (>8), 16 accumulators to better fill SIMD lanes
#if WK_VECTOR_WIDTH <= 8 || WK_COMPLEX
        for (ulong y = 0; y < BLOCK_SIZE; y += 2) {
            for (ulong x = 0; x < BLOCK_SIZE; x += 2) {
#if WK_VECTOR_WIDTH == 1

#if WK_COMPLEX
                wk C11 = {0, 0};
                wk C12 = {0, 0};
                wk C21 = {0, 0};
                wk C22 = {0, 0};
#else
                wk C11 = 0;
                wk C12 = 0;
                wk C21 = 0;
                wk C22 = 0;
#endif

#else
                wk C11 = (wk)(0);
                wk C12 = (wk)(0);
                wk C21 = (wk)(0);
                wk C22 = (wk)(0);
#endif

                __attribute__((opencl_unroll_hint))
                for (ulong z = 0; z < BLOCK_SIZE; z += 2) {
                    // A is row-major in A_tmp: row y at offset y*BLOCK_SIZE
                    base_index = y * BLOCK_SIZE + z;
                    const wk A11 = A_tmp_buffer[base_index];
                    const wk A12 = A_tmp_buffer[base_index + 1];
                    const wk A21 = A_tmp_buffer[base_index + BLOCK_SIZE];
                    const wk A22 = A_tmp_buffer[base_index + BLOCK_SIZE + 1];

                    // B is transposed in B_tmp: "row x" holds column x of original B
                    base_index = x * BLOCK_SIZE + z;
                    const wk B11 = B_tmp_buffer[base_index];
                    const wk B21 = B_tmp_buffer[base_index + 1];
                    const wk B12 = B_tmp_buffer[base_index + BLOCK_SIZE];
                    const wk B22 = B_tmp_buffer[base_index + BLOCK_SIZE + 1];

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

                // Accumulate 2x2 result into the full C tile
                base_index = y * BLOCK_SIZE + x;
#if WK_COMPLEX
                wk prev_value = C_tmp_buffer[base_index];
                prev_value.real += C11.real; prev_value.imag += C11.imag;
                C_tmp_buffer[base_index] = prev_value;

                prev_value = C_tmp_buffer[base_index + 1];
                prev_value.real += C12.real; prev_value.imag += C12.imag;
                C_tmp_buffer[base_index + 1] = prev_value;

                prev_value = C_tmp_buffer[base_index + BLOCK_SIZE];
                prev_value.real += C21.real; prev_value.imag += C21.imag;
                C_tmp_buffer[base_index + BLOCK_SIZE] = prev_value;
                
                prev_value = C_tmp_buffer[base_index + BLOCK_SIZE + 1];
                prev_value.real += C22.real; prev_value.imag += C22.imag;
                C_tmp_buffer[base_index + BLOCK_SIZE + 1] = prev_value;
#else
                C_tmp_buffer[base_index] += C11;
                C_tmp_buffer[base_index + 1] += C12;
                C_tmp_buffer[base_index + BLOCK_SIZE] += C21;
                C_tmp_buffer[base_index + BLOCK_SIZE + 1] += C22;
#endif
            }
        }
        // 4x4 micro-kernel: 16 accumulators for WK_VECTOR_WIDTH > 8.
        // With 16-wide vectors, a 4x4 micro-kernel does 4*4*4 = 64 MADs per
        // inner iteration, giving better arithmetic intensity per register load.
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

/**
 * =============================================================================
 * GEMM PACK — Reorganize matrix from row-major to tile layout
 * =============================================================================
 *
 * Packs a source matrix into a tiled format for efficient GEMM computation.
 * Each tile is a contiguous BLOCK_SIZE x BLOCK_SIZE block stored sequentially
 * in memory. When TRANSPOSE is set, the source is read transposed during
 * packing, so the packed result is already in the correct orientation.
 *
 * COMPILE-TIME PARAMETERS
 * -----------------------
 * BLOCK_SIZE       - Tile dimension (e.g., 2, 4, 8, 16, 32, 64)
 * WK_VECTOR_WIDTH  - Vector width for element addressing
 * TRANSPOSE        - 0: normal copy, 1: transpose during packing
 *
 * KERNEL PARAMETERS
 * -----------------
 * src              - Source matrix in row-major layout (__global, read-only)
 * dst              - Destination buffer in tile layout (__global, write-only)
 * src_row_pitch    - Number of elements per row in the source matrix
 * dst_slice_pitch  - Stride between tile groups along the outer axis
 * dst_row_pitch    - Stride between consecutive k-tiles within a tile group
 * src_rows         - Number of rows in the source matrix (for bounds checking)
 * src_cols         - Number of columns in the source matrix (for bounds checking)
 *
 * NDRANGE (3D)
 * ------------
 * dim 0 (i)  - Tile index along the row/outer axis
 * dim 1 (j)  - Tile index along the k/inner axis
 * dim 2 (k)  - Element index within the tile (flattened BLOCK_SIZE x BLOCK_SIZE)
 *
 * ALGORITHM
 * ---------
 * 1. Compute destination index: i * dst_slice_pitch + j * dst_row_pitch + k
 * 2. Decompose k into (tile_row, tile_col) within the BLOCK_SIZE tile
 * 3. If tile_row >= BLOCK_SIZE, early-exit (padding work-items)
 * 4. Map (tile_row, tile_col) to source coordinates:
 *    - TRANSPOSE=0: src_row = i*BLOCK_SIZE + tile_row, src_col = j*BLOCK_SIZE*VW + tile_col
 *    - TRANSPOSE=1: src_row = j*BLOCK_SIZE*VW + tile_col, src_col = i*BLOCK_SIZE + tile_row
 * 5. Bounds-check against src_rows/src_cols (needed when dimensions are not
 *    a multiple of BLOCK_SIZE, so edge tiles may reference out-of-bounds elements)
 * 6. Copy: dst[dst_index] = src[src_row * src_row_pitch + src_col]
 *
 * =============================================================================
 */

#include "wekua.h"

__kernel void pack(
    __global const wks *const restrict src,
    __global wks *const restrict dst,

    const ulong src_row_pitch,
    const ulong dst_slice_pitch,
    const ulong dst_row_pitch,

    const ulong src_rows,
    const ulong src_cols
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

    const ulong dst_base = i * dst_slice_pitch + j * dst_row_pitch + k;

    // Decompose flat element index k into 2D coordinates within the tile.
    // tile_col runs across columns (width = BLOCK_SIZE * WK_VECTOR_WIDTH),
    // tile_row runs down rows (height = BLOCK_SIZE).
    const ulong tile_col = k % (BLOCK_SIZE * WK_VECTOR_WIDTH);
    const ulong tile_row = (k - tile_col) / (BLOCK_SIZE * WK_VECTOR_WIDTH);
    // NDRange may be padded beyond BLOCK_SIZE rows; discard excess work-items
    if (tile_row >= BLOCK_SIZE) {
        return;
    }

    // Map tile-local coordinates to source matrix coordinates.
    // TRANSPOSE=0: tiles partition the matrix naturally (rows from i, cols from j).
    // TRANSPOSE=1: rows and columns are swapped, so the packed result stores
    //              the transpose without needing a separate transpose pass.
#if TRANSPOSE == 0
    const ulong src_row = i * BLOCK_SIZE + tile_row;
    const ulong src_col = j * (BLOCK_SIZE * WK_VECTOR_WIDTH) + tile_col;
#else
    const ulong src_row = j * (BLOCK_SIZE * WK_VECTOR_WIDTH) + tile_col;
    const ulong src_col = i * BLOCK_SIZE + tile_row;
#endif
    // Edge tiles may extend beyond the actual matrix dimensions
    if (src_col >= src_cols || src_row >= src_rows) {
        return;
    }

    const ulong src_base = src_row * src_row_pitch + src_col;

    dst[dst_base] = src[src_base];
}

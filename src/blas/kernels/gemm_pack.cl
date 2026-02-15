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

    const ulong tile_col = k % (BLOCK_SIZE * WK_VECTOR_WIDTH);
    const ulong tile_row = (k - tile_col) / (BLOCK_SIZE * WK_VECTOR_WIDTH);
    if (tile_row >= BLOCK_SIZE) {
        return;
    }

#if TRANSPOSE == 0
    const ulong src_row = i * BLOCK_SIZE + tile_row;
    const ulong src_col = j * (BLOCK_SIZE * WK_VECTOR_WIDTH) + tile_col;
#else
    const ulong src_row = j * BLOCK_SIZE + tile_col;
    const ulong src_col = i * (BLOCK_SIZE * WK_VECTOR_WIDTH) + tile_row;
#endif
    if (src_col >= src_cols || src_row >= src_rows) {
        return;
    }

    const ulong src_base = src_row * src_row_pitch + src_col;

    dst[dst_base] = src[src_base];
}

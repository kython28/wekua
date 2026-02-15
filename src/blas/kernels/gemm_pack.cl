#include "wekua.h"

__kernel void pack(
    __global const wk *const restrict src,
    __global wk *const restrict dst,

    const ulong src_row_pitch,
    const ulong dst_slice_pitch,
    const ulong dst_row_pitch
) {
    const ulong tile_row = get_global_id(0);
    const ulong tile_col = get_global_id(1);

    const ulong dst_base = tile_row * dst_slice_pitch + tile_col * dst_row_pitch;

#if TRANSPOSE == 0
    ulong src_base = tile_row * BLOCK_SIZE * src_row_pitch + tile_col * BLOCK_SIZE;

    __attribute__((opencl_unroll_hint))
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
        __attribute__((opencl_unroll_hint))
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
            dst[dst_base + y * BLOCK_SIZE + x] = src[src_base + x];
        }
        src_base += src_row_pitch;
    }
#else
    __attribute__((opencl_unroll_hint))
    for (ulong y = 0; y < BLOCK_SIZE; y += 1) {
        ulong src_idx = tile_row * BLOCK_SIZE * src_row_pitch + tile_col * BLOCK_SIZE + y;

        __attribute__((opencl_unroll_hint))
        for (ulong x = 0; x < BLOCK_SIZE; x += 1) {
            dst[dst_base + y * BLOCK_SIZE + x] = src[src_idx];
            src_idx += src_row_pitch;
        }
    }
#endif
}

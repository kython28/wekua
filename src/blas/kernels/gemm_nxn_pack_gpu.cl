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
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    const ulong li = get_local_id(0);
    const ulong lj = get_local_id(1);

    const ulong A_packed_depth = (i - i % BLOCK_SIZE) / BLOCK_SIZE;
    const ulong B_packed_depth = (j - j % BLOCK_SIZE) / BLOCK_SIZE;

    local wk A_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    local wk B_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));

    const ulong local_tile_index = li * BLOCK_SIZE + lj;
    ulong A_base_index = A_packed_depth * A_slice_pitch + local_tile_index;
    ulong B_base_index = B_packed_depth * B_slice_pitch + local_tile_index;

#if WK_VECTOR_WIDTH == 1
    wk C_acc = 0;
#else
    wk C_acc = (wk)(0);
#endif

    for (ulong k = 0; k < cols; k += 1) {
        A_tmp_buffer[local_tile_index] = A_packed[A_base_index];
        B_tmp_buffer[local_tile_index] = B_packed[B_base_index];

        barrier(CLK_LOCAL_MEM_FENCE);

        ulong A_tile_base_index = li * BLOCK_SIZE;
        ulong B_tile_base_index = lj * BLOCK_SIZE;

        __attribute__((opencl_unroll_hint))
        for (ulong kk = 0; kk < BLOCK_SIZE; kk += 1) {
            C_cc += A_tmp_buffer[A_tile_base_index + kk] * B_tmp_buffer[B_tile_base_index + kk];
        }

        A_base_index += A_row_pitch;
        B_base_index += B_row_pitch;
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    ulong C_base = i * C_row_pitch + j;
#if WK_VECTOR_WIDTH == 1
#if HAS_ALPHA
#if HAS_BETA
    C[C_base] = alpha * C_acc + beta * C[C_base];
#else
    C[C_base] = alpha * C_acc;
#endif
#else
    C[C_base] = C_acc;
#endif
#else
#if HAS_ALPHA
#if HAS_BETA
    C[C_base] = alpha * sum(C_acc) + beta * C[C_base];
#else
    C[C_base] = alpha * sum(C_acc);
#endif
#else
    C[C_base] = sum(C_acc);
#endif
#endif
}

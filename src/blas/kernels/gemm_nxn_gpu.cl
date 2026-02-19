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
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    const ulong li = get_local_id(0);
    const ulong lj = get_local_id(1);

    local wk A_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));
    local wk B_tmp_buffer[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(WK_CACHE_LINE_SIZE)));


    const ulong A_local_tile_index = li * BLOCK_SIZE + lj;
    const ulong B_local_tile_index = lj * BLOCK_SIZE + li;
#if WK_VECTOR_WIDTH == 1
    wk C_acc = 0;
#else
    wk C_acc = (wk)(0);
#endif

    for (ulong k = 0; k < cols; k += BLOCK_SIZE) {
#if A_TRANS
        A_tmp_buffer[A_local_tile_index] = A[(k + lj) * A_row_pitch + i];
#else
        A_tmp_buffer[A_local_tile_index] = A[i * A_row_pitch + k + lj];
#endif

#if B_TRANS
        B_tmp_buffer[B_local_tile_index] = B[j * B_row_pitch + k + lj];
#else
        B_tmp_buffer[B_local_tile_index] = B[(k + lj) * B_row_pitch + j];
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        ulong A_base_index = li * BLOCK_SIZE;
        ulong B_base_index = lj * BLOCK_SIZE;
        __attribute__((opencl_unroll_hint))
        for (ulong kk = 0; kk < BLOCK_SIZE; kk += 1) {
            C_cc += A_tmp_buffer[A_base_index + kk] * B_tmp_buffer[B_base_index + kk];
        }

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

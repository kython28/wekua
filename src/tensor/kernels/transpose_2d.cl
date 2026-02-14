#include "wekua.h"

__kernel void transpose_2d(
    __global wks *const restrict A,

    const ulong A_row_pitch,
    const ulong AT_row_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    const ulong index_A = i*A_row_pitch + j;
    const ulong index_AT = j*AT_row_pitch + i;

    const wks value1 = A[index_A];
    const wks value2 = A[index_AT];

    A[index_A] = value2;
    A[index_AT] = value1;
}

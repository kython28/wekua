#include "wekua.h"

__kernel void fill(
    __global wks *restrict buffer,

    const ulong row_pitch,
    const ulong slice_pitch,

    const wks scalar
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

    const ulong index = i*slice_pitch + j*row_pitch + k;
    buffer[index] = scalar;
}

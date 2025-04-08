#include "wekua.h"

__kernel void bias(
    __global wk *const restrict output,
    __constant const wk *const restrict bias,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong col = k << 1;
    const ulong index = i * slice_pitch + j * row_pitch + col;

    output[index] += bias[col];
    output[index + 1] += bias[col + 1];
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    output[index] += bias[k];
#endif
}

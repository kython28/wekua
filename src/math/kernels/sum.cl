#include "wekua.h"

__kernel void sum_kernel(
    __global wk *const restrict x,
    __global wk *const restrict y,

    const ulong row_pitch,
    const ulong slice_pitch,
    const ulong cols
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    const ulong index = i * slice_pitch + j * row_pitch;

#if WK_COMPLEX
    result[index] += x[index];
#else

#if WK_VECTOR_WIDTH == 1
    wk res = 0;
#else
    wk res = (wk)(0);
#endif

    for (ulong k = 0; k < cols; k++) {
        res += x[index + k];
    }

    y[k] = res;
#endif
}

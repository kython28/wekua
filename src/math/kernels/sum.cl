#include "wekua.h"

__kernel void sum_kernel(
    __constant const wk *const restrict x,
    __global wk *const restrict y,

    const ulong row_pitch,
    const ulong slice_pitch,

    const ulong row_pitch2
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    const ulong index = i * slice_pitch + j * row_pitch;

#if WK_VECTOR_WIDTH == 1
    wk res = 0;
#else
    wk res = (wk)(0);
#endif

#if WK_COMPLEX

#if WK_VECTOR_WIDTH == 1
    wk ires = 0;
#else
    wk ires = (wk)(0);
#endif

    for (ulong k = 0; k < row_pitch; k += 2) {
        res += x[index + k];
        ires += x[index + k + 1];
    }

    const ulong index2 = (i * slice_pitch + j) << 1;

    y[index2] = res;
    y[index2 + 1] = ires;
#else
    for (ulong k = 0; k < row_pitch; k++) {
        res += x[index + k];
    }

    y[i * row_pitch2 + j] = res;
#endif
}

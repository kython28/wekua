#include "wekua.h"

__kernel void sum_kernel(
    __constant const wk *const restrict x,
    __global wks *const restrict y,

    const ulong row_pitch,
    const ulong slice_pitch,

    const ulong row_pitch2
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    const ulong index = i * slice_pitch + j * row_pitch;

#if WK_COMPLEX
    wk res = {0, 0};
    for (ulong k = 0; k < row_pitch; k++) {
        const wk val = x[index + k];
        res.real += val.real;
        res.imag += val.imag;
    }
    y[i * row_pitch2 + j] = res;
#else

#if WK_VECTOR_WIDTH == 1
    wk res = 0;
#else
    wk res = (wk)(0);
#endif

    for (ulong k = 0; k < row_pitch; k++) {
        res += x[index + k];
    }

#if WK_VECTOR_WIDTH == 1
    y[i * row_pitch2 + j] = res;
#else
    y[i * row_pitch2 + j] = sum(res);
#endif

#endif
}

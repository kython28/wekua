#include "wekua.h"

__kernel void dot_kernel(
    __global wk *const restrict x,
    __constant const wk *const restrict y,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    wk real_value = x[index];
    wk imag_value = x[index + 1];

    COMPLEX_MUL_K(wk)
    COMPLEX_MUL(real_value, imag_value, y[index], y[index + 1]);

    x[index] = real_value;
    x[index + 1] = imag_value;
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    x[index] *= y[index];
#endif
}

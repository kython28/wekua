#include "wekua.h"

__kernel void tanh_dev(
    __constant wk *const restrict input,
    __global wk *const restrict derivative,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    wk real_value = input[index];
    wk imag_value = input[index + 1];

    COMPLEX_MUL_K(wk)
    COMPLEX_MUL(real_value, imag_value, real_value, imag_value);

#if dtype == 9
    input[index] = 1.0 - real_value;
#else
    input[index] = 1.0f - real_value;
#endif

    input[index + 1] = -imag_value;

#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    const wk value = input[index];
#if dtype == 9
    input[index] = 1.0 - value*value;
#else
    input[index] = 1.0f - value*value;
#endif

#endif
}

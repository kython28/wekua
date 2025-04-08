#include "wekua.h"

__kernel void mse_kernel(
    __constant const wk *const restrict output,
    __constant const wk *const restrict expected,
    __global wk *const restrict error_tensor,

#if CALC_DEV
    __global wk *const restrict dev_output,
#endif

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);


#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    wk error = expected[index] - output[index];
    wk errori = expected[index + 1] - output[index + 1];
#if CALC_DEV

#if WK_DTYPE == 9
    dev_output[index] = -2.0 * error;
    dev_output[index + 1] = -2.0 * errori;
#elif WK_DTYPE == 8
    dev_output[index] = -2.0f * error;
    dev_output[index + 1] = -2.0f * errori;
#else
    dev_output[index] = -2*error;
    dev_output[index + 1] = -2*errori;
#endif

#endif

    COMPLEX_MUL_K(wk)
    COMPLEX_MUL(error, errori, error, errori);

    error_tensor[index] = error;
    error_tensor[index + 1] = errori;
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    wk error = expected[index] - output[index];
    error_tensor[index] = error*error;

#if CALC_DEV

#if WK_DTYPE == 9
    dev_output[index] = -2.0 * error;
#elif WK_DTYPE == 8
    dev_output[index] = -2.0f * error;
#else
    dev_output[index] = -2 * error;
#endif

#endif

#endif

}

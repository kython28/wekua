#include "wekua.h"

__kernel void mse_kernel(
    __constant const wk *const restrict output,
    __constant const wk *const restrict expected,
    __global wk *const restrict error_tensor

#if CALC_DEV
    , __global wk *const restrict dev_output
#endif
) {
    const ulong index = get_global_id(0);

#if WK_COMPLEX
    wk o = output[index];
    wk e = expected[index];
    wk err = (wk){ e.real - o.real, e.imag - o.imag };

#if CALC_DEV
    dev_output[index] = (wk){ -(wks)2 * err.real, -(wks)2 * err.imag };
#endif

    wk squared;
    COMPLEX_MUL(err, err, squared);
    error_tensor[index] = squared;
#else
    wk err = expected[index] - output[index];
    error_tensor[index] = err*err;

#if CALC_DEV
    dev_output[index] = -(wks)2 * err;
#endif

#endif
}

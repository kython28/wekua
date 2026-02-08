#include "wekua.h"

__kernel void sigmoid(
    __global wk *const restrict output,
    const ulong number_of_elements
) {
	const ulong index = get_global_id(0);
    if (index >= number_of_elements) return;

#if WK_COMPLEX
    wk val = output[index];
    const wks denominator = (wks)1 + exp(-val.real);

    val = (wk){ ((wks)1 / denominator) * cos(val.imag), -((wks)1 / denominator) * sin(val.imag) };
    output[index] = val;
#else
    const wk exp_value = exp(-output[index]);
    output[index] = (wks)1 / ((wks)1 + exp_value);
#endif
}

__kernel void sigmoid_dev(
    __constant const wk *const restrict output,
    __global wk *const restrict derivative,
    const ulong number_of_elements
) {
    const ulong index = get_global_id(0);
    if (index >= number_of_elements) return;

#if WK_COMPLEX
    wk val = output[index];
    wk one_minus = (wk){ (wks)1 - val.real, -val.imag };

    wk result;
    COMPLEX_MUL(val, one_minus, result);

    derivative[index] = result;
#else
    const wk value = output[index];
    derivative[index] = value * ((wks)1 - value);
#endif
}

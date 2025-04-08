#include "wekua.h"

__kernel void sigmoid(
    __global wk *const restrict output,

    const ulong row_pitch,
    const ulong slice_pitch
) {
	const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    const wk real_value = output[index];
    const wk imag_value = output[index + 1];

    const wk denominator = 1 + exp(-real_value);

#if dtype == 9
    output[index] = (1.0 / denominator) * cos(imag_value);
    output[index + 1] = -(1.0 / denominator) * sin(imag_value);
#else
    output[index] = (1.0f / denominator) * cos(imag_value);
    output[index + 1] = -(1.0f / denominator) * sin(imag_value);
#endif

#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    const wk exp_value = exp(-output[index]);

#if dtype == 9
    output[index] = 1.0 / (1.0 + exp_value);
#else
    output[index] = 1.0f / (1.0f + exp_value);
#endif

#endif

}

__kernel void sigmoid_dev(
    __constant const wk *const restrict output,
    __global wk *const restrict derivative,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    wk real_value = output[index];
    wk imag_value = output[index + 1];

#if dtype == 9
    const wk value = 1.0 - real_value;
#else
    const wk value = 1.0f - real_value;
#endif

    COMPLEX_MUL_K(wk)
    COMPLEX_MUL(real_value, imag_value, value, -imag_value);

    derivative[index] = real_value;
    derivative[index + 1] = imag_value;
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    const wk value = output[index];
#if dtype == 9
    derivative[index] = value * (1.0 - value);
#else
    derivative[index] = value * (1.0f - value);
#endif

#endif
}

#include "wekua.h"

__kernel void sin_kernel(
    __global wk *const restrict input,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    const wk real_value = input[index];
    const wk imag_value = input[index + 1];

    input[index] = sin(real_value) * cosh(imag_value);
    input[index + 1] = cos(real_value) * sinh(imag_value);
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    input[index] = sin(input[index]);
#endif
}

__kernel void cos_kernel(
    __global wk *const restrict input,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    const wk real_value = input[index];
    const wk imag_value = input[index + 1];

    input[index] = cos(real_value) * cosh(imag_value);
    input[index + 1] = -sin(real_value) * sinh(imag_value);
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    input[index] = cos(input[index]);
#endif
}

__kernel void tan_kernel(
    __global wk *const restrict input,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    const wk real_value = input[index];
    const wk imag_value = input[index + 1];

    input[index] = (sin(real_value) * cos(real_value) + sinh(imag_value) * cosh(imag_value)) / (cos(real_value) * cosh(imag_value));
    input[index + 1] = (sinh(imag_value) * cos(real_value) - sin(real_value) * cosh(imag_value)) / (cos(real_value) * cosh(imag_value));
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    input[index] = tan(input[index]);
#endif
}

__kernel void sinh_kernel(
    __global wk *const restrict input,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    const wk real_value = input[index];
    const wk imag_value = input[index + 1];

    input[index] = sinh(real_value) * cos(imag_value);
    input[index + 1] = cosh(real_value) * sin(imag_value);
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    input[index] = sinh(input[index]);
#endif
}

__kernel void cosh_kernel(
    __global wk *const restrict input,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    const wk real_value = input[index];
    const wk imag_value = input[index + 1];

    input[index] = cosh(real_value) * cos(imag_value);
    input[index + 1] = sinh(real_value) * sin(imag_value);
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    input[index] = cosh(input[index]);
#endif
}

__kernel void tanh_kernel(
    __global wk *const restrict input,

    const ulong row_pitch,
    const ulong slice_pitch
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i * slice_pitch + j * row_pitch + (k << 1);

    const wk real_value = input[index];
    const wk imag_value = input[index + 1];

    const wk denominator = cosh(real_value) * cos(imag_value);

    input[index] = (sinh(real_value) * cosh(real_value) + sin(imag_value) * cos(imag_value)) / denominator;
    input[index + 1] = (sin(imag_value) * cosh(real_value) - sinh(real_value) * cos(imag_value)) / denominator;
    
#else
    const ulong index = i * slice_pitch + j * row_pitch + k;

    input[index] = tanh(input[index]);
#endif
}

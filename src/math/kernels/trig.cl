#include "wekua.h"

__kernel void sin_kernel(__global wk *const restrict input) {
    const ulong index = get_global_id(0);
#if WK_COMPLEX
    const wk val = input[index];
    input[index] = (wk){ sin(val.real) * cosh(val.imag), cos(val.real) * sinh(val.imag) };
#else
    input[index] = sin(input[index]);
#endif
}

__kernel void cos_kernel(__global wk *const restrict input) {
    const ulong index = get_global_id(0);
#if WK_COMPLEX
    const wk val = input[index];
    input[index] = (wk){ cos(val.real) * cosh(val.imag), -sin(val.real) * sinh(val.imag) };
#else
    input[index] = cos(input[index]);
#endif
}

__kernel void tan_kernel(__global wk *const restrict input) {
    const ulong index = get_global_id(0);
#if WK_COMPLEX
    const wk val = input[index];
    const T two_a = 2 * val.real;
    const T two_b = 2 * val.imag;
    const T denom = cos(two_a) + cosh(two_b);
    input[index] = (wk){ sin(two_a) / denom, sinh(two_b) / denom };
#else
    input[index] = tan(input[index]);
#endif
}

__kernel void sinh_kernel(__global wk *const restrict input) {
    const ulong index = get_global_id(0);
#if WK_COMPLEX
    const wk val = input[index];
    input[index] = (wk){ sinh(val.real) * cos(val.imag), cosh(val.real) * sin(val.imag) };
#else
    input[index] = sinh(input[index]);
#endif
}

__kernel void cosh_kernel(__global wk *const restrict input) {
    const ulong index = get_global_id(0);
#if WK_COMPLEX
    const wk val = input[index];
    input[index] = (wk){ cosh(val.real) * cos(val.imag), sinh(val.real) * sin(val.imag) };
#else
    input[index] = cosh(input[index]);
#endif
}

__kernel void tanh_kernel(__global wk *const restrict input) {
    const ulong index = get_global_id(0);
#if WK_COMPLEX
    const wk val = input[index];
    const T two_a = 2 * val.real;
    const T two_b = 2 * val.imag;
    const T denom = cosh(two_a) + cos(two_b);
    input[index] = (wk){ sinh(two_a) / denom, sin(two_b) / denom };
#else
    input[index] = tanh(input[index]);
#endif
}

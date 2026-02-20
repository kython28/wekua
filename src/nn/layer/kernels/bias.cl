#include "wekua.h"

__kernel void bias(
    __global wk *const restrict output,
    __global const wk *const restrict bias,
    const ulong row_pitch
) {
    const ulong index = get_global_id(0);

    const ulong col = index % row_pitch;

#if WK_COMPLEX
    wk o = output[index];
    wk b = bias[col];
    output[index] = (wk){ o.real + b.real, o.imag + b.imag };
#else
    output[index] += bias[col];
#endif
}

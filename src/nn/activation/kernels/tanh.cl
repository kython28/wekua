#include "wekua.h"

__kernel void tanh_dev(
    __constant const wk *const restrict input,
    __global wk *const restrict derivative,
    const ulong number_of_elements
) {
    const ulong index = get_global_id(0);
    if (index >= number_of_elements) return;

#if WK_COMPLEX
    wk val = input[index];

    wk squared;
    COMPLEX_MUL(val, val, squared);

    derivative[index] = (wk){ (wks)1 - squared.real, -squared.imag };
#else
    const wk value = input[index];
    derivative[index] = (wks)1 - value*value;
#endif
}

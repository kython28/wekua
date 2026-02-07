#include "wekua.h"

__kernel void dot_kernel(
    __global wk *const restrict x,
    __constant const wk *const restrict y
) {
    const ulong index = get_global_id(0);
#if WK_COMPLEX
    const wk a = x[index];
    const wk b = y[index];
    COMPLEX_MUL_K(T)
    COMPLEX_MUL(a, b, x[index]);
#else
    x[index] *= y[index];
#endif
}

#include "wekua.h"

__kernel void dot_kernel(
    __global wk *const restrict x,
    __global const wk *const restrict y

#if WK_VECTOR_WIDTH == 1
    , const ulong x_slice_pitch,
    const ulong x_row_pitch,

    const ulong y_slice_pitch,
    const ulong y_row_pitch
#endif
) {
#if WK_VECTOR_WIDTH > 1
    const ulong x_index = get_global_id(0);
    const ulong y_index = x_index;
#else
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

    const ulong x_index = i * x_slice_pitch + j * x_row_pitch + k;
    const ulong y_index = i * y_slice_pitch + j * y_row_pitch + k;
#endif

#if WK_COMPLEX
    const wk a = x[x_index];
    const wk b = y[y_index];
    COMPLEX_MUL_K(T)
    COMPLEX_MUL(a, b, x[x_index]);
#else
    x[x_index] *= y[y_index];
#endif
}

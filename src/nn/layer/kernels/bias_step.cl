#include "wekua.h"

__kernel void bias_step(
    __constant const wk *const restrict derivative,
    __global wk *const restrict bias_gradients,
    const ulong dev_row_pitch,
    const ulong dev_rows
) {

#if WK_COMPLEX
    const ulong i = get_global_id(0);

    wk res = (wk){ 0, 0 };

    ulong base = i;
    for (ulong r = 0; r < dev_rows; r++, base += dev_row_pitch) {
        wk d = derivative[base];
        res = (wk){ res.real + d.real, res.imag + d.imag };
    }

    bias_gradients[i] = res;
#else

#if WK_VECTOR_WIDTH == 1
    wk res = 0;
#else
    wk res = (wk)(0);
#endif

    const ulong i = get_global_id(0);

    ulong base = i;
    for (ulong r = 0; r < dev_rows; r++, base += dev_row_pitch) {
        res += derivative[base];
    }

    bias_gradients[i] = res;
#endif
}

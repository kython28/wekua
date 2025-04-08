#include "wekua.h"

__kernel void bias_step(
    __constant const wk *restrict const derivative,
    __global wk *restrict const bias_gradients,

    const ulong dev_row_pitch,
    const ulong dev_rows
) {

#if WK_VECTOR_WIDTH == 1
    wk res = 0;
#else
    wk res = (wk)(0);
#endif

#if WK_COMPLEX
    const ulong i = get_global_id(0) << 1;

#if WK_VECTOR_WIDTH == 1
    wk ires = 0;
#else
    wk ires = (wk)(0);
#endif

    ulong base = i;
    for (ulong r = 0; r < dev_rows; r++, base += dev_row_pitch) {
        res += derivative[base];
        ires += derivative[base + 1];
    }

    bias_gradients[i] = res;
    bias_gradients[i + 1] = ires;
#else
    const ulong i = get_global_id(0);

    ulong base = i;
    for (ulong r = 0; r < dev_rows; r++, base += dev_row_pitch) {
        res += derivative[base];
    }

    bias_gradients[i] = res;
#endif
} 

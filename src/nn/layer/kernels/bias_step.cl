#include "wekua.h"

__kernel void bias_step(
    __constant const wk *restrict const derivative,
    __global wk *restrict const bias_gradients,

    const ulong row_pitch,
    const ulong slice_pitch,

	unsigned long col
) {
    const ulong i = get_global_id(0) << 1;
    const ulong j = get_global_id(1) << 1;

#if WK_VECTOR_WIDTH == 1
    wk C11 = 0;
    wk C12 = 0;
#else
    wk C11 = (wk)(0);
    wk C12 = (wk)(0);
#endif

    const ulong index = i * row_pitch + j * slice_pitch;

#if WK_COMPLEX

#if WK_VECTOR_WIDTH == 1
    wk C11i = 0;
    wk C12i = 0;
#else
    wk C11i = (wk)(0);
    wk C12i = (wk)(0);
#endif

    for (ulong k = 0; k < col; k += 4) {
        ulong base = index + k;
        C11 += derivative[base];
        C11i += derivative[base + 1];

        C12 += derivative[base + 2];
        C12i += derivative[base + 3];

        base += row_pitch;
        C11 += derivative[base];
        C11i += derivative[base + 1];

        C12 += derivative[base + 2];
        C12i += derivative[base + 3];
    }

    const ulong index2 = (j << 1);
    bias_gradients[j] = C11;
    bias_gradients[j + 1] = C11i;
    bias_gradients[j + 2] = C12;
    bias_gradients[j + 3] = C12i;
#else
    for (ulong k = 0; k < col; k += 2) {
        const ulong base = index + k;
        C11 += derivative[base] + derivative[base + 1];
        C12 += derivative[base + row_pitch] + derivative[base + row_pitch + 1];
    }

    bias_gradients[j] = C11;
    bias_gradients[j + 1] = C12;
#endif
} 

#include "wekua.h"

__kernel void bias_step(
    __constant const wk *restrict const bias,
    __global wk *restrict const derivative,

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
        C11 += bias[index + k];
        C11i += bias[index + k + 1];

        C12 += bias[index + k + 2];
        C12i += bias[index + k + 3];

        C11 += bias[index + col + k];
        C11i += bias[index + col + k + 1];

        C12 += bias[index + col + k + 2];
        C12i += bias[index + col + k + 3];
    }

    derivative[index] = C11;
    derivative[index + 1] = C11;
    derivative[index + 1] = C12;
    derivative[index + 2] = C12;
#else
    for (ulong k = 0; k < col; k += 2) {
        C11 += bias[index + k] + bias[index + k + 1];
        C12 += bias[index + col + k] + bias[index + col + k + 1];
    }

    derivative[index] = C11;
    derivative[index + 1] = C12;
#endif
} 

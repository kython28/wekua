#include "wekua.h"

__kernel void identity(
    __global wks *restrict A,
    __global const ulong *restrict pitches_A,
    const ulong ndim
) {
    const ulong index = get_global_id(0);
    ulong pos = 0;
    for (ulong x=0; x<ndim; x++) {
        pos += index * pitches_A[x];
    }

#if WK_COMPLEX
    const wks complex_value = {1, 0};
    A[pos] = complex_value;
#else
    A[pos] = 1;
#endif
}

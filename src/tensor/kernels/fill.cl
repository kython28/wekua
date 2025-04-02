#include "wekua.h"

__kernel void fill(
    __global wks *restrict buffer,

    const ulong row_pitch,
    const ulong slice_pitch,

    const wks real_scalar
#if WK_COMPLEX
    , const wks imag_scalar
#endif
) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i*slice_pitch + j*row_pitch + (k << 1);

    buffer[index] = real_scalar;
    buffer[index + 1] = imag_scalar;
#else
    const ulong index = i*slice_pitch + j*row_pitch + k;

    buffer[index] = real_scalar;
#endif
}

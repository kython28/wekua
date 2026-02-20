#include "wekua.h"

__kernel void to_complex(
	__global const wks *const restrict src,
    __global wks *const restrict dst,

	const ulong src_row_pitch,
    const ulong src_slice_pitch,

    const ulong dst_row_pitch,
    const ulong dst_slice_pitch
) {
	const ulong i = get_global_id(0);
	const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

	const ulong dst_index = ( i * dst_slice_pitch + j * dst_row_pitch + k ) << 1;
	dst[dst_index + OFFSET] = src[i * src_slice_pitch + j * src_row_pitch + k];
}

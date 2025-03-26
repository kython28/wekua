#include "wekua.h"

#ifndef OFFSET
#define	OFFSET 0
#endif

__kernel void to_complex(
	__global const wks *restrict src,
    __global wks *restrict dst,

	const ulong row_pitch1,
    const ulong row_pitch2
) {
	const ulong i = get_global_id(0);
	const ulong j = get_global_id(1);

	const dst_index = i * row_pitch2 + (j << 1);
	dst[dst_index + OFFSET] = src[i * row_pitch1 + j];
	dst[dst_index + (1 - OFFSET)] = 0;
}

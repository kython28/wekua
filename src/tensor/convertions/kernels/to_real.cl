#include "wekua.h"

#ifndef OFFSET
#define	OFFSET 0
#endif

__kernel void to_real(
	__global const wks *restrict src,
    __global wks *restrict dst,

	const ulong row_pitch1,
    const ulong row_pitch2
) {
	const ulong i = get_global_id(0);
	const ulong j = get_global_id(1);

	dst[i * row_pitch2 + j] = src[i * row_pitch1 + (j << 1) + OFFSET];
}

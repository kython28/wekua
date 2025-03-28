#include "wekua.h"

__kernel void transpose(
	__global const wks *restrict A,
    __global const ulong *restrict pitches_A,

	__global wks *restrict B,
    __global const ulong *restrict pitches_B,

    const ulong A_row_pitch,
    const ulong A_slice_pitch,
    const ulong A_height,
    const ulong A_cols,

	const ulong dim0,
    const ulong dim1,
    const ulong ndim
) {
	ulong index = get_global_id(0);
    if ((index % A_row_pitch) >= A_cols) return;
    else if ((index % A_slice_pitch) >= A_height) return;

	const wks value = A[index];

	ulong b_index = 0;
	for (ulong x=0; x<ndim; x++) {
		const ulong dim_pitch = pitches_A[x];
		const ulong remaining = index % dim_pitch;
		const ulong dim_index = (index - remaining) / dim_pitch;

		if (x == dim0) {
			b_index += dim_index * pitches_B[dim1];
		}else if (x == dim1) {
			b_index += dim_index * pitches_B[dim0];
		}else{
			b_index += dim_index * pitches_B[x];
		}

		index = remaining;
	}
	B[b_index + index] = value;
}

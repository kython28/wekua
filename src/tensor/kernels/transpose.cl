#include "wekua.h"

__kernel void transpose(
	__global const wks *A, __global const ulong *pitches_A,
	__global wks *B, __global const ulong *pitches_B,
	const ulong dim0, const ulong dim1, const ulong ndim
) {
	ulong index = get_global_id(0);
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
		index -= dim_index * dim_pitch;
	}
	B[b_index + index] = value;
}

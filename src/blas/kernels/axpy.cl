#include "wekua.h"

__kernel void axpy(__global const wk *A, __global wk *B, const wks alpha) {
	const ulong i = get_global_id(0);

	B[i] += alpha * A[i];
}

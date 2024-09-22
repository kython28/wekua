#include "wekua.h"

__kernel void axpy(
	__global const wk *A, __global wk *B,
	const wks alpha
#if com == 1
	, const wks beta
#endif
) {
	const ulong i = get_global_id(0);

#if com == 1
	const ulong index = (i << 1);
	wks real_value = A[index];
	wks imag_value = A[index + 1];
	COMPLEX_MUL_K(wks)
	COMPLEX_MUL(real_value, imag_value, alpha, beta);

	B[index] += real_value;
	B[index + 1] += imag_value;
#else
	B[i] += alpha * A[i];
#endif
}


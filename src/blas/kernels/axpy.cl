#include "wekua.h"

#define COMPLEX_MUL_SCALAR_SCALAR(a, b, c, d) \
	wks k1, k2, k3; \
	k1 = c*(a + b); \
	k2 = a*(d - c); \
	k3 = b*(c + d); \
	a = k1 - k3; \
	b = k1 + k2; \

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
	COMPLEX_MUL_SCALAR_SCALAR(real_value, imag_value, alpha, beta);

	B[index] += real_value;
	B[index + 1] += imag_value;
#else
	B[i] += alpha * A[i];
#endif
}


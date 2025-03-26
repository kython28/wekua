#include "wekua.h"

__kernel void axpy(
	__global const wk *restrict A,
    __global wk *restrict B,
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

__kernel void axpy2(
	__global const wks *restrict A,
    __global wks *restrict B,

    const ulong row_pitch_A,
    const ulong row_pitch_B,

	const wks alpha
#if com == 1
	, const wks beta
#endif
) {
	const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

#if com == 1
    const col = j << 1;
    const ulong index_A = i * row_pitch_A + col;
    const ulong index_B = i * row_pitch_B + col;

	wks real_value = A[index_A];
	wks imag_value = A[index_A + 1];

	COMPLEX_MUL_K(wks)
	COMPLEX_MUL(real_value, imag_value, alpha, beta);

	B[index_B] += real_value;
	B[index_B + 1] += imag_value;
#else
    const ulong index_A = i * row_pitch_A + j;
    const ulong index_B = i * row_pitch_B + j;

	B[index_B] += alpha * A[index_A];
#endif
}

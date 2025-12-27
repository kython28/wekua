#include "wekua.h"

__kernel void axpy(
	__global const wk *const restrict A,
    __global wk *const restrict B,

    const ulong A_row_pitch,
    const ulong A_slice_pitch
#if WK_COMPLEX == 0 && WK_VECTOR_WIDTH == 1
    , const ulong B_row_pitch,
    const ulong B_slice_pitch
#endif

#if HAS_ALPHA
	, const wks alpha
#endif
) {
	const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX == 0 && WK_VECTOR_WIDTH == 1
    const ulong A_index = i * A_slice_pitch + j * A_row_pitch + k;
    const ulong B_index = i * B_slice_pitch + j * B_row_pitch + k;
#else
    const ulong A_index = i * A_slice_pitch + j * A_row_pitch + k;
    const ulong B_index = A_index;
#endif

#if WK_COMPLEX == 1
    wk complex_a_value = A[A_index];
    wk complex_b_value = B[B_index];

#if HAS_ALPHA
	COMPLEX_MUL_K(T)
	COMPLEX_MUL(complex_a_value, alpha, complex_a_value);

    complex_b_value.real += complex_a_value.real;
    complex_b_value.imag += complex_a_value.imag;

	B[B_index] = complex_b_value;
#else

#if SUBSTRACT
    complex_b_value.real -= complex_a_value.real;
    complex_b_value.imag -= complex_a_value.imag;
    B[B_index] = complex_b_value;
#else
    complex_b_value.real += complex_a_value.real;
    complex_b_value.imag += complex_a_value.imag;
    B[B_index] = complex_b_value;
#endif

#endif

#else

#if HAS_ALPHA
	B[B_index] += alpha * A[A_index];
#else

#if SUBSTRACT
    B[B_index] -= A[A_index];
#else
    B[B_index] += A[A_index];
#endif

#endif

#endif
}

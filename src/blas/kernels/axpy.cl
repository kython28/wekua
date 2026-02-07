#include "wekua.h"

__kernel void axpy(
	__global const wk *const restrict A,
    __global wk *const restrict B
#if HAS_ALPHA
	, const wks alpha
#endif
) {
	const ulong index = get_global_id(0);

#if WK_COMPLEX == 1
    wk complex_a_value = A[index];
    wk complex_b_value = B[index];

#if HAS_ALPHA
	COMPLEX_MUL_K(T)
	COMPLEX_MUL(complex_a_value, alpha, complex_a_value);

    complex_b_value.real += complex_a_value.real;
    complex_b_value.imag += complex_a_value.imag;

	B[index] = complex_b_value;
#else

#if SUBSTRACT
    complex_b_value.real -= complex_a_value.real;
    complex_b_value.imag -= complex_a_value.imag;
    B[index] = complex_b_value;
#else
    complex_b_value.real += complex_a_value.real;
    complex_b_value.imag += complex_a_value.imag;
    B[index] = complex_b_value;
#endif

#endif

#else

#if HAS_ALPHA
	B[index] += alpha * A[index];
#else

#if SUBSTRACT
    B[index] -= A[index];
#else
    B[index] += A[index];
#endif

#endif

#endif
}

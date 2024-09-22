#include "wekua.h"

__kernel void gemm(
	__global const wk *A, __global const wk *B,
	__global wk *C, const ulong a_slice, const ulong a_pitch,
	const b_slice, const ulong b_pitch, const ulong c_slice,
	const ulong c_pitch, const ulong k_lenght, const wks alpha,
#if beta_is_zero == 0
	, const wks beta
#endif
#if com == 1
	, const wks alphai,
#if beta_is_zero == 0
	, const wks betai
#endif
#endif
) {
	const ulong i = get_global_id(0);
	const ulong j = get_global_id(1);
	const ulong n = get_global_id(2);

	const ulong a_matrix_index = n * a_slice;
	const ulong b_matrix_index = n * b_slice;

#if a_trans == 0
	const ulong a_row = a_matrix_index + i * a_pitch;
#endif

#if b_trans == 1
	const ulong b_row = b_matrix_index + j * b_pitch;
#endif
	const c_index = n * c_slice + i * c_pitch + j;

#if com == 1

#else
	#if optimized_gemm == 1
		#if wk_width == 1
		wk C11 = 0;
		wk C12 = 0;
		wk C21 = 0;
		wk C22 = 0;
		#else
		wk C11 = (wk)(0);
		wk C12 = (wk)(0);
		wk C21 = (wk)(0);
		wk C22 = (wk)(0);
		#endif

		for (ulong k=0; k<k_lenght; k++) {
		#if a_trans == 0
			const ulong a_index = a_row + k;
			wk A11 = A[a_index];
			wk A12 = A[a_index + 1];
			wk A21 = A[a_index + a_pitch];
			wk A22 = A[a_index + a_pitch + 1];
		#elif a_trans == 1
			const ulong a_index = k * a_pitch + i + a_matrix_index;
			wk A11 = A[a_index];
			wk A21 = A[a_index + 1];
			wk A12 = A[a_index + a_pitch];
			wk A22 = A[a_index + a_pitch + 1];
		#endif

		#if b_trans == 0
			const ulong b_index = k * b_pitch + j + b_matrix_index;
			wk B11 = B[b_index];
			wk B12 = B[b_index + 1];
			wk B21 = B[b_index + b_pitch];
			wk B22 = B[b_index + b_pitch + 1];
		#elif b_trans == 1
			const ulong b_index = b_row + k;
			wk B11 = B[b_index];
			wk B21 = B[b_index + 1];
			wk B12 = B[b_index + b_pitch];
			wk B22 = B[b_index + b_pitch + 1];
		#endif

			const wk t0 = B22 - B12;
			const wk t1 = B11 + t0;
			const wk t2 = A11 - A21;
			const wk t3 = t2 - A22;

			const wk m0 = A11 * B11;
			const wk m1 = A12 * B21;
			const wk m2 = A22 * (t1 - B21);
			const wk m3 = t2 * t0;
			const wk m4 = (A21 + A22) * (B12 - B11);
			const wk m5 = (t3 + A12) * B22;
			const wk m6 = t3 * t1;

			const wk p0 = m0 - m6;
			const wk p1 = p0 + m3;

			C11 += m0 + m1;
			C12 += p0 + m4 + m5;
			C21 += p1 - m2;
			C22 += p1 + m4;
		}

		#if beta_is_zero == 1
			C[c_index] = alpha * C11;
			C[c_index + 1] = alpha * C12;
			C[c_index + c_pitch] = alpha * C21;
			C[c_index + c_pitch + 1] = alpha * C22;
		#else
			C[c_index] = alpha * C11 + beta * C[c_index];
			C[c_index + 1] = alpha * C12 + beta * C[c_index + 1];
			C[c_index + c_pitch] = alpha * C21 + beta * C[c_index + c_pitch];
			C[c_index + c_pitch + 1] = alpha * C22 + beta * C[c_index + c_pitch + 1];
		#endif
	#else
		wk result = 0;
		for (ulong k=0; k<k_lenght; k++) {
		#if a_trans == 0 and b_trans == 0
			result += A[a_row + k] * B[k * b_pitch + j + b_matrix_index];
		#elif a_trans == 1 and b_trans == 0
			result += A[k * a_pitch + i + a_matrix_index] * B[k * b_pitch + j + b_matrix_index];
		#elif a_trans == 0 and b_trans == 1
			result += A[a_row + k] * B[b_row + k];
		#elif a_trans == 1 and b_trans == 1
			result += A[k * a_pitch + i + a_matrix_index] * B[b_row + k];
		#endif
		}

		C[c_index] = alpha * result + beta * C[c_index];
#endif

#endif
}

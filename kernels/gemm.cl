#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void gemm( // Winograd
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	__global wks *cr, __global wks *ci,

	wks ralpha, wks ialpha, wks rbeta,
	wks ibeta,

	unsigned long col, unsigned long col2,
	unsigned char com
){
	unsigned long i = get_global_id(0) << 1;
	unsigned long j = get_global_id(1) << 1;

	unsigned long row_a, row_a1;
	unsigned long row_b, row_b1;
	unsigned long row_c, row_c1;

	row_a = i*col2;
	row_a1 = row_a + col2;

	row_b = j*col2;
	row_b1 = row_b + col2;

	row_c = i*col + j;
	row_c1 = row_c + col;

	wk C11, C12, C21, C22;

	#if width == 1
	C11 = 0;
	C12 = 0;
	C21 = 0;
	C22 = 0;
	#else
	C11 = (wk)(0);
	C12 = (wk)(0);
	C21 = (wk)(0);
	C22 = (wk)(0);
	#endif

	if (com){
		wk C11_i, C12_i, C21_i, C22_i;

		#if width == 1
		C11_i = 0;
		C12_i = 0;
		C21_i = 0;
		C22_i = 0;
		#else
		C11_i = (wk)(0);
		C12_i = (wk)(0);
		C21_i = (wk)(0);
		C22_i = (wk)(0);
		#endif

		for (unsigned long k=0; k<col2; k += 2){
			wk A11, A12, A21, A22;
			wk A11_i, A12_i, A21_i, A22_i;

			wk B11, B12, B21, B22;
			wk B11_i, B12_i, B21_i, B22_i;

			A11 = ar[row_a + k];
			A12 = ar[row_a + k + 1];
			A21 = ar[row_a1 + k];
			A22 = ar[row_a1 + k + 1];

			A11_i = ai[row_a + k];
			A12_i = ai[row_a + k + 1];
			A21_i = ai[row_a1 + k];
			A22_i = ai[row_a1 + k + 1];

			B11 = br[row_b + k];
			B21 = br[row_b + k + 1];
			B12 = br[row_b1 + k];
			B22 = br[row_b1 + k + 1];

			B11_i = bi[row_b + k];
			B21_i = bi[row_b + k + 1];
			B12_i = bi[row_b1 + k];
			B22_i = bi[row_b1 + k + 1];

			wk t0, t1, t2, t3;
			wk t0_i, t1_i, t2_i, t3_i;

			wk m0, m1, m2, m3, m4, m5, m6;
			wk m0_i, m1_i, m2_i, m3_i, m4_i, m5_i, m6_i;

			wk p0, p1;
			wk p0_i, p1_i;

			t0 = B22 - B12;
			t1 = B11 + t0;
			t2 = A11 - A21;
			t3 = t2 - A22;

			t0_i = B22_i - B12_i;
			t1_i = B11_i + t0_i;
			t2_i = A11_i - A21_i;
			t3_i = t2_i - A22_i;

			// M0
			m0 = A11; m0_i = A11_i;
			complex_mul(&m0, &m0_i, B11, B11_i);

			// M1
			m1 = A12; m1_i = A12_i;
			complex_mul(&m1, &m1_i, B21, B21_i);

			// M2
			m2 = A22; m2_i = A22_i;
			complex_mul(&m2, &m2_i, t1 - B21, t1_i - B21_i);

			// M3
			m3 = t2; m3_i = t2;
			complex_mul(&m3, &m3_i, t0, t0_i);

			// m4
			m4 = A21 + A22; m4_i = A21_i + A22_i;
			complex_mul(&m4, &m4_i, B12 - B11, B12_i - B11_i);

			// m5
			m5 = t3 + A12; m5_i = t3_i + A12_i;
			complex_mul(&m5, &m5_i, B22, B22_i);

			// m6
			m6 = t3; m6_i = t3_i;
			complex_mul(&m6, &m6_i, t1, t1_i);

			p0 = m0 - m6;
			p1 = p0 + m3;

			p0_i = m0_i - m6_i;
			p1_i = p0_i + m3_i;

			C11 += m0 + m1;
			C12 += p0 + m4 + m5;
			C21 += p1 - m2;
			C22 += p1 + m4;

			C11_i += m0_i + m1_i;
			C12_i += p0_i + m4_i + m5_i;
			C21_i += p1_i + m2_i;
			C22_i += p1_i + m4_i;
		}

		#if width == 1
		ci[row_c] = C11_i; ci[row_c + 1] = C12_i;
		ci[row_c1] = C21_i; ci[row_c1 + 1] = C22_i;
		#else
		ci[row_c] = sum(C11_i);
		ci[row_c + 1] = sum(C12_i);
		ci[row_c1] = sum(C21_i);
		ci[row_c1 + 1] = sum(C22_i);
		#endif

	}else{
		for (unsigned long k=0; k<col2; k += 2){
			wk A11, A12, A21, A22;
			wk B11, B12, B21, B22;

			A11 = ar[row_a + k];
			A12 = ar[row_a + k + 1];
			A21 = ar[row_a1 + k];
			A22 = ar[row_a1 + k + 1];

			B11 = br[row_b + k];
			B21 = br[row_b + k + 1];
			B12 = br[row_b1 + k];
			B22 = br[row_b1 + k + 1];

			wk t0, t1, t2, t3;
			wk m0, m1, m2, m3, m4, m5, m6;
			wk p0, p1;

			t0 = B22 - B12;
			t1 = B11 + t0;
			t2 = A11 - A21;
			t3 = t2 - A22;

			m0 = A11*B11;
			m1 = A12*B21;
			m2 = A22*(t1 - B21);
			m3 = t2*t0;
			m4 = (A21 + A22)*(B12 - B11);
			m5 = (t3 + A12)*B22;
			m6 = t3*t1;

			p0 = m0 - m6;
			p1 = p0 + m3;

			C11 += m0 + m1;
			C12 += p0 + m4 + m5;
			C21 += p1 - m2;
			C22 += p1 + m4;
		}

		#if width == 1
		cr[row_c] = ralpha*C11 + rbeta*cr[row_c];
		cr[row_c + 1] = ralpha*C12 + rbeta*cr[row_c + 1];
		cr[row_c1] = ralpha*C21 + rbeta*cr[row_c1];
		cr[row_c1 + 1] = ralpha*C22 + rbeta*cr[row_c1 + 1];
		#else
		cr[row_c] = ralpha*sum(C11) + rbeta*cr[row_c];
		cr[row_c + 1] = ralpha*sum(C12) + rbeta*cr[row_c + 1];
		cr[row_c1] = ralpha*sum(C21) + rbeta*cr[row_c1];
		cr[row_c1 + 1] = ralpha*sum(C22) + rbeta*cr[row_c1 + 1];
		#endif
	}
}
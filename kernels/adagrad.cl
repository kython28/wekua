#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void adagrad(
	__global wk *err, __global wk *erri,
	__global wk *wr, __global wk *wi,
	__global wk *gr, __global wk *gi,
	wks alpha, wks alphai,
	unsigned long col, unsigned char com
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wk error = err[i], errori, gra = gr[i], grai, k1, k2;
	if (com){
		errori = erri[i];
		grai = gi[i];

		k1 = error;
		k2 = errori;

		complex_mul(&k1, &k2, error, errori);

		gra += k1;
		grai += k2;

		k2 = sqrt(gra*gra + grai*grai);

		#if dtype == 8
		k1 = sqrt((k2 + gra)/2 + FLT_EPSILON);
		k2 = sqrt((k2 - gra)/2 + FLT_EPSILON);
		#else
		k1 = sqrt((k2 + gra)/2 + DBL_EPSILON);
		k2 = sqrt((k2 - gra)/2 + DBL_EPSILON);
		#endif

		calc_inv_complex(&k1, &k2);
		complex_mul_scal(&error, &errori, alpha, alphai);
		complex_mul(&error, &errori, k1, k2);

		wr[i] -= error;
		wi[i] -= errori;

		gi[i] = grai;
	}else{
		gra = gr[i];
		gra += error*error;

		#if dtype == 8
		wr[i] -= alpha*error/sqrt(gra + FLT_EPSILON);
		#else
		wr[i] -= alpha*error/sqrt(gra + DBL_EPSILON);
		#endif
	}
	gr[i] = gra;
}
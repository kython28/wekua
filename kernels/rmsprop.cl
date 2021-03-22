#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void rmsprop(
	__global wk *err, __global wk *erri,
	__global wk *wr, __global wk *wi,
	__global wk *gr, __global wk *gi,
	wks alpha, wks alphai,
	wks beta, wks betai,
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
		complex_mul_scal(&k1, &k2, 1 - beta, betai);
		complex_mul_scal(&gra, &grai, beta, betai);

		gra += k1;
		grai += k2;

		k2 = sqrt(gra*gra + grai*grai);

		k1 = sqrt((k2 + gra)/2 + FLT_EPSILON);
		k2 = sqrt((k2 - gra)/2 + FLT_EPSILON);

		calc_inv_complex(&k1, &k2);
		complex_mul_scal(&error, &errori, alpha, alphai);
		complex_mul(&error, &errori, k1, k2);

		wr[i] -= error;
		wi[i] -= errori;

		gi[i] = grai;
	}else{
		gra = beta*gr[i];
		gra += (1 - beta)*error*error;

		wr[i] -= alpha*error/sqrt(gra + FLT_EPSILON);
	}
	gr[i] = gra;
}
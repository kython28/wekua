#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void adagrad(
	__global wk *err, __global wk *erri,
	__global wk *wr, __global wk *wi,
	__global wk *gr, __global wk *gi,
	wks alpha, wks alphai,
	unsigned long col
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wk error = err[i], gra = gr[i];
#if com
	wk errori, grai, k1, k2;
	errori = erri[i];
	grai = gi[i];

	k1 = error;
	k2 = errori;

	complex_mul(&k1, &k2, error, errori);

	gra += k1;
	grai += k2;

	k2 = sqrt(gra*gra + grai*grai);

	k1 = sqrt((k2 + gra)/2) + FLT_EPSILON;
	k2 = sqrt((k2 - gra)/2) + FLT_EPSILON;

	calc_inv_complex(&k1, &k2);
	complex_mul_scal(&error, &errori, alpha, alphai);
	complex_mul(&error, &errori, k1, k2);

	wr[i] -= error;
	wi[i] -= errori;

	gi[i] = grai;
#else
	gra = gr[i];
	gra += error*error;

	wr[i] -= alpha*error*(rsqrt(gra) + FLT_EPSILON);
#endif
	gr[i] = gra;
}
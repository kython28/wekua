#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void adam(
	__global wk *gra1r, __global wk *gra1i,
	__global wk *gra2r, __global wk *gra2i,
	__global wk *err, __global wk *erri,
	__global wk *wr, __global wk *wi,

	wks alpha, wks alphai,
	wks beta1r, wks beta1i,
	wks beta2r, wks beta2i,

	unsigned long col
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wk t_gra1, t_gra2, error = err[i];
	wk t_m, t_v;
#if com
	wk t_gra1i, t_gra2i, errori = erri[i];
	wk t_mi, t_vi, k1, k2;

	t_gra2 = gra2r[i];
	t_gra2i = gra2i[i];

	t_gra1 = error;
	t_gra1i = errori;
	complex_mul(&t_gra1, &t_gra1i, error, errori);
	complex_mul_scal(&t_gra1, &t_gra1i, 1 - beta2r, -beta2i);
	complex_mul(&t_gra2, &t_gra2i, beta1r, beta1i);

	t_gra2 += t_gra1;
	t_gra2i += t_gra1i;

	t_gra1 = gra1r[i];
	t_gra1i = gra1i[i];

	complex_mul(&t_gra1, &t_gra1i, beta1r, beta1i);
	complex_mul_scal(&error, &errori, 1 - beta1r, -beta1i);

	t_gra1 += error;
	t_gra1i += errori;

	t_m = t_gra1;
	t_mi = t_gra1i;

	beta1r = 1 - beta1r;
	beta1i = -beta1i;

	calc_inv_complex_scal(&beta1r, &beta1i);
	complex_mul_scal(&t_m, &t_mi, beta1r, beta1i);

	beta2r = 1 - beta2r;
	beta2i = -beta2i;

	t_v = t_gra2;
	t_vi = t_gra2i;

	calc_inv_complex_scal(&beta2r, &beta2i);
	complex_mul_scal(&t_v, &t_vi, beta2r, beta2i);

	k2 = sqrt(t_v*t_v, t_vi*t_vi);

	k1 = sqrt((k2 + t_v)/2);
	k2 = sqrt((k2 - t_v)/2);

	calc_inv_complex(&k1, &k2);
	complex_mul_scal(&t_m, &t_mi, alpha, alphai);
	complex_mul(&t_m, &t_mi, k1, k2);

	wr[i] -= t_m;
	wi[i] -= t_mi;

	gra1r[i] = t_gra1;
	gra2r[i] = t_gra2;
	gra1r[i] = t_gra1;
	gra2r[i] = t_gra2;

#else
	t_gra1 = beta1r*gra1r[i] + (1 - beta1r)*error;
	t_gra2 = beta1r*gra2r[i] + (1 - beta2r)*error*error;

	t_m = t_gra1/(1 - beta1r);
	t_v = t_gra2/(1 - beta2r);

	wr[i] -= alpha*t_m/(sqrt(t_v) + FLT_EPSILON);

	gra1r[i] = t_gra1;
	gra2r[i] = t_gra2;
#endif
}
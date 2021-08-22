#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void adadelta(
	__global wk *gra1r, __global wk* gra1i,
	__global wk *gra2r, __global wk *gra2i,
	__global wk *err, __global wk *erri,
	__global wk *wr, __global wk *wi,

	wks alpha, wks alphai,

	unsigned long long col
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wk t_gra1, t_gra2, error = err[i];
	wk k1, k2;
#if com
	wk t_gra1i, t_gra2i, errori;
	errori = erri[i];

	t_gra1 = gra1r[i];
	t_gra1i = gra1i[i];
	complex_mul_scal(&t_gra1, &t_gra1i, alpha, alphai);

	t_gra2 = error;
	t_gra2i = errori;
	complex_mul(&t_gra2, &t_gra2i, error, errori);
	complex_mul_scal(&t_gra2, &t_gra2i, 1 - alpha, -alphai);

	t_gra1 += t_gra2;
	t_gra1i += t_gra2i;

	gra1r[i] = t_gra1;
	gra1i[i] = t_gra1i;

	t_gra2 = gra2r[i];
	t_gra2i = gra2i[i];

	calc_inv_complex(&t_gra1, &t_gra1i);
	complex_mul(&t_gra1, &t_gra1i, t_gra2, t_gra2i);
	t_gra1 += 1;

	k2 = sqrt(t_gra1*t_gra1 + t_gra1i*t_gra1i);

	k1 = sqrt((k2 + t_gra1)/2);
	k2 = sqrt((k2 - t_gra1)/2);

	complex_mul(&error, &errori, k1, k2);
	wr[i] -= error;
	wi[i] -= errori;

	complex_mul_scal(&t_gra2, &t_gra2i, alpha, alphai);
	complex_mul(&error, &errori, error, errori);
	complex_mul_scal(&error, &errori, 1 - alpha, -alphai);

	t_gra2 += error;
	t_gra2i += errori;

	gra2r[i] = t_gra2;
	gra2i[i] = t_gra2i;
#else
	t_gra1 = alpha*gra1r[i];
	t_gra1 += (1 - alpha)*error*error;

	t_gra2 = gra2r[i];

	error *= sqrt(t_gra2/t_gra1 + 1);
	wr[i] -= error;

	t_gra2 *= alpha;
	t_gra2 += (1 - alpha)*error*error;

	gra1r[i] = t_gra1;
	gra2r[i] = t_gra2;
#endif
}
#include "/usr/lib/wekua_kernels/dtype.cl"

#if dtype == 8
#define EPSILON 1e-7
#else
#define EPSILON 1e-14
#endif

void calc_poly(wks r, wks i, wks *a, wks *b, __global wks *rpoly, __global wks *ipoly, unsigned long col){
	a[0] = 0.0;
	b[0] = 0.0;
	wks c = 1.0, d = 0.0, aa = 0.0, bb = 0.0;
	for (unsigned long x=0; x<col; x++){
		aa += rpoly[x]*c - ipoly[x]*d;
		bb += rpoly[x]*d + ipoly[x]*c;
		complex_mul_scal2(&c, &d, r, i);
	}
	a[0] = aa; b[0] = bb;
}

void calc_ratio(wks r, wks i, wks *fr, wks *fi, __global wks *rpoly, __global wks *ipoly, __global wks *rdev, __global wks *idev, unsigned long col){
	wks dr, di;

	calc_poly(r, i, fr, fi, rpoly, ipoly, col+1);
	calc_poly(r, i, &dr, &di, rdev, idev, col);

	calc_inv_complex_scal(&dr, &di);

	complex_mul_scal2(fr, fi, dr, di);
}

__kernel void aberth(
	__global wks *rroot, __global wks *iroot,
	__global wks *rpoly, __global wks *ipoly,
	__global wks *rdev, __global wks *idev,
	unsigned long col
){
	unsigned long i = get_global_id(0);

	wks ratior, ratioi, devr, devi;
	wks ro, io, error = DBL_MAX;

	while (error > EPSILON){
		calc_ratio(rroot[i], iroot[i], &ratior, &ratioi, rpoly, ipoly, rdev, idev, col);

		devr = 0.0; devi = 0.0;
		for (unsigned long x=0; x < col; x++){
			if (x != i){
				ro = rroot[i]; io = iroot[i];
				ro -= rroot[x]; io -= iroot[x];

				calc_inv_complex_scal(&ro, &io);
				devr += ro; devi += io;
			}
		}

		complex_mul_scal2(&devr, &devi, ratior, ratioi);
		devr = 1.0 - devr;
		devi *= -1.0;
		calc_inv_complex_scal(&devr, &devi);
		complex_mul_scal2(&ratior, &ratioi, devr, devi);

		rroot[i] -= ratior;
		iroot[i] -= ratioi;

		error = sqrt(ratior*ratior + ratioi*ratioi);
	}
}
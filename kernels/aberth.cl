#include "/usr/lib/wekua_kernels/dtype.cl"

#if dtype == 8
#define EPSILON 1e-7
#else
#define EPSILON 1e-14
#endif

void complex_mul(wks *a, wks *b, wks c, wks d){
	wks e, f, g, h;
	g = a[0]; h = b[0];
	e = g*c - h*d;
	f = g*d + h*c;
	a[0] = e;
	b[0] = f;
}

void calc_inv_complex(wks *a, wks *b){
	wks c, d, aa, bb;
	aa = a[0]; bb = b[0];

	c = aa;
	d = -1.0*bb;

	aa = (aa*aa + bb*bb);

	c /= aa;
	d /= aa;
	
	a[0] = c;
	b[0] = d;
}

void calc_poly(wks r, wks i, wks *a, wks *b, __global wks *rpoly, __global wks *ipoly, unsigned long col){
	a[0] = 0.0;
	b[0] = 0.0;
	wks c = 1.0, d = 0.0, aa = 0.0, bb = 0.0;
	for (unsigned long x=0; x<col; x++){
		aa += rpoly[x]*c - ipoly[x]*d;
		bb += rpoly[x]*d + ipoly[x]*c;
		complex_mul(&c, &d, r, i);
	}
	a[0] = aa; b[0] = bb;
}

void calc_ratio(wks r, wks i, wks *fr, wks *fi, __global wks *rpoly, __global wks *ipoly, __global wks *rdev, __global wks *idev, unsigned long col){
	wks dr, di;

	calc_poly(r, i, fr, fi, rpoly, ipoly, col+1);
	calc_poly(r, i, &dr, &di, rdev, idev, col);

	calc_inv_complex(&dr, &di);

	complex_mul(fr, fi, dr, di);
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

				calc_inv_complex(&ro, &io);
				devr += ro; devi += io;
			}
		}

		complex_mul(&devr, &devi, ratior, ratioi);
		devr = 1.0 - devr;
		devi *= -1.0;
		calc_inv_complex(&devr, &devi);
		complex_mul(&ratior, &ratioi, devr, devi);

		rroot[i] -= ratior;
		iroot[i] -= ratioi;

		error = sqrt(ratior*ratior + ratioi*ratioi);
	}
}
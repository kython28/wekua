void complex_mul(double *a, double *b, double c, double d){
	double e, f;
	e = a[0]*c - b[0]*d;
	f = a[0]*d + b[0]*c;
	a[0] = e;
	b[0] = f;
}

void calc_poly(double r, double i, double *a, double *b, __global double *rpoly, __global double *ipoly, unsigned int col){
	a[0] = 0.0;
	b[0] = 0.0;
	double c = 1.0, d = 0.0;
	for (unsigned int x=0; x<col; x++){
		a[0] += rpoly[x]*c - ipoly[x]*d;
		b[0] += rpoly[x]*d + ipoly[x]*c;
		complex_mul(&c, &d, r, i);
	}
}

void calc_inv_complex(double *a, double *b){
	double c, d;
	c = a[0]/(a[0]*a[0]+b[0]*b[0]);
	d = -1.0*b[0]/(a[0]*a[0]+b[0]*b[0]);
	a[0] = c;
	b[0] = d;
}

void calc_ratio(double r, double i, double *fr, double *fi, __global double *rpoly, __global double *ipoly, __global double *rdev, __global double *idev, unsigned int col){
	double dr, di;

	calc_poly(r, i, fr, fi, rpoly, ipoly, col+1);
	calc_poly(r, i, &dr, &di, rdev, idev, col);

	calc_inv_complex(&dr, &di);

	complex_mul(fr, fi, dr, di);
}

__kernel void aberth(__global double *rroot, __global double *iroot,
	__global double *rpoly, __global double *ipoly,
	__global double *rdev, __global double *idev,
	unsigned int col){
	unsigned int i = get_global_id(0);

	double ratior, ratioi, devr, devi;
	double ro, io, error = DBL_MAX;

	while (error > 1e-14){
		calc_ratio(rroot[i], iroot[i], &ratior, &ratioi, rpoly, ipoly, rdev, idev, col);

		devr = 0.0; devi = 0.0;
		for (unsigned int x=0; x < col; x++){
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
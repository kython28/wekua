void complex_mul(double *a, double *b, double c, double d){
	double e, f, g, h;
	g = a[0]; h = b[0];
	e = g*c - h*d;
	f = g*d + h*c;
	a[0] = e;
	b[0] = f;
}

void step_one(double *a, double *b, double r, double h, double y){
	double c = cosh(h) - sinh(h);
	a[0] = cos(y*r)*c;
	b[0] = sin(y*r)*c;
}

void step_two(double *a, double *b, double h, double r, double x){
	double er, co, si, mwo, awo;
	er = exp(r);
	co = cos(h)*er; si = sin(h)*er;

	awo = atan2(si, co);
	mwo = pow(sqrt(co*co + si*si), x);
	a[0] = mwo*cos(awo*x);
	b[0] = mwo*sin(awo*x);
}

__kernel void power(__global double *a, __global double *b,
	double alpha, double beta, unsigned char com){
	unsigned long i = get_global_id(0);
	double aa, bb, r, h, so, soi;
	if (com){
		aa = a[i]; bb = b[i];
		r = 0.5*log(aa*aa + bb*bb);
		h = atan2(bb,aa);

		step_one(&so, &soi, r, h, beta);
		step_two(&aa, &bb, r, h, alpha);

		complex_mul(&aa, &bb, so, soi);

		a[i] = aa;
		b[i] = bb;
	}else{
		a[i] = pow(a[i], alpha);
	}
}
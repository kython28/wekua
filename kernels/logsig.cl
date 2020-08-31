void complex_mul(__global double *a, __global double *b, double c, double d){
	double e, f;
	e = a[0]*c - b[0]*d;
	f = a[0]*d + b[0]*c;
	a[0] = e;
	b[0] = f;
}

void calc_inv_complex(double *a, double *b){
	double c, d;
	c = a[0]/(a[0]*a[0]+b[0]*b[0]);
	d = -1.0*b[0]/(a[0]*a[0]+b[0]*b[0]);
	a[0] = c;
	b[0] = d;
}

__kernel void logsig(__global double *a, __global double *b,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

	double c,d;
	c = exp(-1.0*a[i]);
	if (com){
		d = c*sin(b[i]);
		c *= cos(b[i]);

		a[i] = c;
		b[i] = d;
		c += 1.0;

		calc_inv_complex(&c, &d);
		complex_mul(&a[i], &b[i], c, d);
	}else{
		a[i] = 1.0/(c+1.0);
	}
}
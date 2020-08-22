void calc_coeff(double a, double b, __global double *c, __global double *d){
	double r, ang, cc, dd;
	cc = c[0]; dd = d[0];

	r = a/(a*a+b*b);
	ang = -1.0*b/(a*a+b*b);

	a = cc*r - dd*ang;
	b = cc*ang + dd*r;

	c[0] = a;
	d[0] = b;
}

__kernel void gauss2(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	double cc, dd;
	if (com){
		calc_coeff(a[i*col+i], b[i*col+i], &c[i*col+j], &d[i*col+j]);
	}else{
		c[i*col+j] /= a[i*col+i];
	}
}
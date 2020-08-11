void calc_coeff(double a, double b, __global double *c, __global double *d){
	double r, ang;

	r = a/(a*a+b*b);
	ang = -1.0*b/(a*a+b*b);

	a = c[0]*r - d[0]*ang;
	b = c[0]*ang + d[0]*r;

	c[0] = a;
	d[0] = b;
}

__kernel void gauss2(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned char com){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);

	double cc, dd;
	if (com){
		calc_coeff(a[i*col+i], b[i*col+i], &c[i*col+j], &d[i*col+j]);
	}else{
		c[i*col+j] /= a[i*col+i];
	}
}
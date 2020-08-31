void complex_mul(__global double *a, __global double *b, double c, double d){
	double e, f, g, h;
	g = a[0]; h = b[0];
	e = g*c - h*d;
	f = g*d + h*c;
	a[0] = e;
	b[0] = f;
}

void calc_inv_complex(double *a, double *b){
	double c, d, aa, bb;
	aa = a[0]; bb = b[0];
	c = aa/(aa*aa+bb*bb);
	d = -1.0*bb/(aa*aa+bb*bb);
	a[0] = c;
	b[0] = d;
}

__kernel void divide(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned char com, unsigned long col, unsigned long col2,
	unsigned long offsetar, unsigned long offsetac,
	unsigned long offsetbr, unsigned long offsetbc){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	double aa, bb;
	if (com){
		aa = c[(i+offsetbr)*col2+j+offsetbc]; bb = d[(i+offsetbr)*col2+j+offsetbc];
		calc_inv_complex(&aa, &bb);
		complex_mul(&a[(i+offsetar)*col+j+offsetac], &b[(i+offsetar)*col+j+offsetac], aa, bb);
	}else{
		a[(i+offsetar)*col+j+offsetac] /= c[(i+offsetbr)*col2+j+offsetbc];
	}
}
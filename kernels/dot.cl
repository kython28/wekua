void complex_mul(__global double *a, __global double *b, double c, double d){
	double e, f, g, h;
	g = a[0]; h = b[0];
	e = g*c - h*d;
	f = g*d + h*c;
	a[0] = e;
	b[0] = f;
}

__kernel void dotm(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned char com, unsigned long col,
	unsigned long col2, unsigned long offsetar, unsigned long offsetac,
	unsigned long offsetbr, unsigned long offsetbc){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);
	if (com){
		complex_mul(&a[(i+offsetar)*col+j+offsetac], &b[(i+offsetar)*col+j+offsetac], c[(i+offsetbr)*col2+j+offsetbc], d[(i+offsetbr)*col2+j+offsetbc]);
	}else{
		a[(i+offsetar)*col+j+offsetac] *= c[(i+offsetbr)*col2+j+offsetbc];
	}
}
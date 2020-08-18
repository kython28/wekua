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
	unsigned char com){
	unsigned long i = get_global_id(0);
	if (com){
		complex_mul(&a[i], &b[i], c[i], d[i]);
	}else{
		a[i] *= c[i];
	}
}
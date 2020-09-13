__kernel void coseh(__global double *a, __global double *b,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	double aa, bb;
	if (com){
		aa = a[i];
		bb = b[i];
		a[i] = cosh(aa)*cos(bb);
		b[i] = -1.0*sinh(aa)*sin(bb);
	}else{
		a[i] = cos(a[i]);
	}
}
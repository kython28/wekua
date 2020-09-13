__kernel void senh(__global double *a, __global double *b,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	double aa, bb;
	if (com){
		aa = a[i];
		bb = b[i];
		a[i] = sinh(aa)*cos(bb);
		b[i] = cosh(aa)*sin(bb);
	}else{
		a[i] = sinh(a[i]);
	}
}
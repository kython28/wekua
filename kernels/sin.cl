__kernel void sen(__global double *a, __global double *b, unsigned char com){
	unsigned long i = get_global_id(0);
	double aa, bb;
	if (com){
		aa = a[i];
		bb = b[i];
		a[i] = sin(aa)*cosh(bb);
		b[i] = cos(aa)*sinh(bb);
	}else{
		a[i] = sin(a[i]);
	}
}
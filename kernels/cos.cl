__kernel void cose(__global double *a, __global double *b, unsigned char com){
	unsigned long i = get_global_id(0);
	double aa, bb;
	if (com){
		aa = a[i];
		bb = b[i];
		a[i] = cos(aa)*cosh(bb);
		b[i] = -1.0*sin(aa)*sinh(bb);
	}else{
		a[i] = cos(a[i]);
	}
}
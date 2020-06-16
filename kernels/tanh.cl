__kernel void tgh(__global double *a, __global double *b, unsigned char com){
	unsigned long i = get_global_id(0);
	double c;
	if (com){
		c = cosh(a[i]*2) + cos(b[i]*2);
		a[i] = sinh(a[i]*2)/c;
		b[i] = sin(b[i]*2)/c;
	}else{
		a[i] = tanh(a[i]);
	}
}
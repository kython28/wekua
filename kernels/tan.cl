__kernel void tg(__global double *a, __global double *b, unsigned char com){
	unsigned long i = get_global_id(0);
	double c;
	if (com){
		c = cos(a[i]*2) + cosh(b[i]*2);
		a[i] = sin(a[i]*2)/c;
		b[i] = sinh(b[i]*2)/c;
	}else{
		a[i] = tan(a[i]);
	}
}
__kernel void softplus(__global double *a, __global double *b,
	unsigned char com){
	unsigned long i = get_global_id(0);
	double c,d;
	c = exp(a[i]);
	if (com){
		d = c*sin(b[i]);
		c *= cos(b[i]);

		a[i] = c + 1.0;
		b[i] = d;
	}else{
		a[i] = c + 1.0;
	}
}
__kernel void lognatu(__global double *a, __global double *b,
	unsigned char com){
	unsigned long i = get_global_id(0);

	double n, m;
	if (com){
		n = a[i];
		m = b[i];
		b[i] = atan2(m, n);
		a[i] = 0.5*log(n*n + m*m);
	}else{
		a[i] = log(a[i]);
	}
}
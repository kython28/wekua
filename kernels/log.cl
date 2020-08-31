__kernel void lognatu(__global double *a, __global double *b,
	unsigned char com, unsigned long col){
	unsigned long current = get_global_id(0)*col+get_global_id(1);
	double n, m;
	if (com){
		n = a[current];
		m = b[current];
		b[current] = atan2(m, n);
		a[current] = 0.5*log(n*n + m*m);
	}else{
		a[current] = log(a[current]);
	}
}
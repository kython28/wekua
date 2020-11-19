#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void lognatu(__global wk *a, __global wk *b,
	unsigned char com, unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	wk n, m;
	if (com){
		n = a[i];
		m = b[i];
		b[i] = atan2(m, n);
		a[i] = 0.5*log(n*n + m*m);
	}else{
		a[i] = log(a[i]);
	}
}
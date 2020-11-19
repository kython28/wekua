#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void axpy(__global wk *a, __global wk *b,
	__global wk *c, __global wk *d,
	wks alpha, wks beta, unsigned char com){
	unsigned long i = get_global_id(0);

	if (com){
		c[i] += alpha*a[i] - beta*b[i];
		d[i] += alpha*b[i] + beta*a[i];
	}else{
		c[i] += alpha*a[i];
	}
}
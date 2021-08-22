#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void doth(__global wk *a, __global wk *b,
	__global wk *c, __global wk *d, unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

#if com
	wk aa, bb, cc, dd;
	aa = a[i]; bb = b[i];
	cc = c[i]; dd = d[i];

	a[i] = aa*cc - bb*dd;
	b[i] = aa*dd + bb*cc;
#else
	a[i] *= c[i];
#endif
}
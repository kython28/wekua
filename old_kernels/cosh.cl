#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void coseh(__global wk *a, __global wk *b,
	unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
#if com
	wk aa, bb;
	aa = a[i];
	bb = b[i];
	a[i] = cosh(aa)*cos(bb);
	b[i] = -1.0*sinh(aa)*sin(bb);
#else
	a[i] = cos(a[i]);
#endif
}
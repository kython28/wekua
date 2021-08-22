#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void tgh(__global wk *a, __global wk *b,
	unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

#if com
	wk c = cosh(a[i]*2) + cos(b[i]*2);
	a[i] = sinh(a[i]*2)/c;
	b[i] = sin(b[i]*2)/c;
#else
	a[i] = tanh(a[i]);
#endif
}
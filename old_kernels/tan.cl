#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void tg(__global wk *a, __global wk *b,
	unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
#if com
	wk c = cos(a[i]*2) + cosh(b[i]*2);;
	a[i] = sin(a[i]*2)/c;
	b[i] = sinh(b[i]*2)/c;
#else
	a[i] = tan(a[i]);
#endif
}
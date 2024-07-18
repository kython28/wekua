#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void uniform(__global wk *a, __global wk *b,
	wks ra, wks ia, wks re, wks ie){
	unsigned long i = get_global_id(0);

#if com
#if wk_width == 1
		b[i] = ia + (ie-ia)*b[i];
#else
		b[i] = (wk)(ia) + (ie-ia)*b[i];
#endif
#else
#if wk_width == 1
	a[i] = ra + (re-ra)*a[i];
#else
	a[i] = (wk)(ra) + (re-ra)*a[i];
#endif
#endif
}
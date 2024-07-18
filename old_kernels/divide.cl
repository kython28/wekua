#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void divide(__global wks *a, __global wks *b,
	__global wks *c, __global wks *d, unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
#if com
	wks aa, bb, cc, dd;
	aa = c[i]; bb = d[i];

	cc = aa*aa + bb*bb;

	#if dtype >= 8
	if (cc == 0){
		#if dtype == 8
		cc += FLT_EPSILON;
		#else
		cc += DBL_EPSILON;
		#endif
	}
	#endif

	dd = -1.0*bb/cc;
	cc = aa/cc;

	aa = a[i]; bb = b[i];

	a[i] = aa*cc - bb*dd;
	b[i] = aa*dd + bb*cc;
#else
	a[i] /= c[i];
#endif
}
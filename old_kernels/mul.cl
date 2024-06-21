#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void mul(__global wks *a, __global wks *b,
	__global wks *c, __global wks *d,
	unsigned long col, unsigned long rcol,
	unsigned long rcol2
){
	unsigned long i = get_local_id(0);
	unsigned long j = i*rcol;
	i *= rcol2;

	wks aa, bb, cc, dd, ee, ff;
#if com
	cc = a[j]; dd = a[j];
	for (unsigned long x=1; x<col; x++){
		aa = a[j+x]; bb = b[j+x];
		ee = cc; ff = dd;

		cc = aa*ee - bb*ff;
		dd = aa*ff + bb*ee;
	}
	c[i] = cc;
	d[i] = dd;
#else
	aa = a[j];
	for (unsigned long x=1; x<col; x++){
		aa *= a[j+x];
	}
	c[i] = aa;
#endif
}
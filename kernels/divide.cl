#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void divide(__global wks *a, __global wks *b,
	__global wks *c, __global wks *d, unsigned long col,
	unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

	wks aa, bb, cc, dd;
	if (com){
		aa = c[i]; dd = d[i];

		cc = aa/(aa*aa + bb*bb);
		dd = -1.0*bb/(aa*aa + bb*bb);

		aa = a[i]; bb = b[i];

		a[i] = aa*cc - bb*dd;
		b[i] = aa*dd + bb*cc;
	}else{
		a[i] /= c[i];
	}
}
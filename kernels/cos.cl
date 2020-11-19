#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void cose(__global wk *a, __global wk *b,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	wk aa, bb;
	if (com){
		aa = a[i];
		bb = b[i];
		a[i] = cos(aa)*cosh(bb);
		b[i] = -1.0*sin(aa)*sinh(bb);
	}else{
		a[i] = cos(a[i]);
	}
}
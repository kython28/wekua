#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void sen(__global wk *a, __global wk *b,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	wk aa, bb;
	if (com){
		aa = a[i];
		bb = b[i];
		a[i] = sin(aa)*cosh(bb);
		b[i] = cos(aa)*sinh(bb);
	}else{
		a[i] = sin(a[i]);
	}
}
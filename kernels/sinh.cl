#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void senh(__global wk *a, __global wk *b,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	wk aa, bb;
	if (com){
		aa = a[i];
		bb = b[i];
		a[i] = sinh(aa)*cos(bb);
		b[i] = cosh(aa)*sin(bb);
	}else{
		a[i] = sinh(a[i]);
	}
}
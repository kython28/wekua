#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void absolute(__global wk *a, __global wk *b,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

	#if dtype >= 8
	wk aa, bb;
	#endif
	if (com){
		#if dtype >= 8
		aa = a[i]; bb = b[i];
		a[i] = sqrt(aa*aa + bb*bb);
		#endif
	}else{
	#if dtype >= 8
		a[i] = fabs(a[i]);
	#else
		a[i] = abs(a[i]);
	#endif
	}
}
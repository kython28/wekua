#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void sqrt_kernel(
	__global wk *ar, __global wk *ai
){
	unsigned long i = get_global_id(0);

#if com
	wk aa, bb, r;
	aa = ar[i]; bb = ai[i];

	r = sqrt(aa*aa + bb*bb);

	ar[i] = sqrt((r+aa)/2);
	ai[i] = sqrt((r-aa)/2);
#else
	ar[i] = sqrt(ar[i]);
#endif
}
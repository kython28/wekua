#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void fill(
	__global wks *ar, __global wks *ai,
	wks alpha, wks beta, unsigned long col
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	ar[i] = alpha;
#if com
	ai[i] = beta;
#endif
}
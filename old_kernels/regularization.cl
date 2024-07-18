#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void regularization(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	ulong col
){
	ulong i = get_global_id(0);
	ulong j = get_global_id(1);

	i = i*col + j;

	ar[i] += br[j];
#if com
	ai[i] += bi[j];
#endif
}
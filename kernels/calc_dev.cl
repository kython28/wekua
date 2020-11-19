#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void calc_dev(
	__global wks *ar, __global wks *ai,
	__global wks *br, __global wks *bi,
	unsigned char com
){
	unsigned long i = get_global_id(0) + 1;

	ar[i-1] = i*br[i];
	if (com) ai[i-1] = i*bi[i];
}
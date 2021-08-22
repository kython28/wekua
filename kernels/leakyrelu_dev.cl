#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void leakyrelu_dev(
	__global wks *ar,
	__global wks *br, __global wks *bi,
	wks alpha, wks alphai,
	unsigned long col
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wks aa = ar[i];
	if (aa < 0.0){
		br[i] = alpha;
#if com
		bi[i] = alphai;
#else
	}else if (aa > 0.0){
		br[i] = 1.0;
	}
}
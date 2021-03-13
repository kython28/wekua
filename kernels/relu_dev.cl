#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void relu_dev(
	__global wks *ar, __global wks *br,
	unsigned long col
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	if (ar[i] <= 0.0){
		br[i] = 0.0;
	}else{
		br[i] = 1.0;
	}
}
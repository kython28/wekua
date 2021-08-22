#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void relu(
	__global wks *ar, __global wks *ai,
	unsigned long col
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	if (ar[i] < 0.0){
		ar[i] = 0.0;
#if com
		ai[i] = 0.0;
#endif
	}
}
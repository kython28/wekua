#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void relu(
	__global wk *ar, __global wk *ai,
	unsined char com
){
	unsigned long i = get_global_id(0);

	if (ar[i] < 0.0){
		ar[i] = 0.0;
		if (com){
			ai[i] = 0.0;
		}
	}
}
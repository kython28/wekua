#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void conv1d(
	__global wks *inputr, __global wks *inputi,
	__global wks *outputr, __global wks *outputi
	__global wks *wr, __global wks *wi,
	unsigned long col
){
	unsigned long x = get_global_id(0);
	unsigned long y = get_global_id(1);

	unsigned long stop = col-1;
	for (unsigned long i=0; i<stop; i++){

	}
}
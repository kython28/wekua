#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void scalar_add(
	__global wk *real, __global wk *imag,
	wks real_k, wks imag_k,
	unsigned long col
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);
	real[i] += real_k;
#if com
	imag[i] += imag_k;
#endif
}
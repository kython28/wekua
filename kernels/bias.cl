#include "/usr/lib/wekua_kernels/dtype.cl"


__kernel void bias(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	unsigned long col, unsigned char com

	#if mem_type == 0x1
	, __local wk *atiler, __local wk *atilei
	#endif
){
	unsigned long j = get_global_id(1);
	unsigned long i = get_global_id(0)*col + j;

	#if mem_type == 0x1
	unsigned long y = get_local_id(0);
	unsigned long x = get_local_id(1);

	if (y == 0){
		atiler[x] = ar[j];
		if (com){
			atilei[x] = ai[j];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	br[i] += atiler[x];

	if (com){
		br[i] += atilei[x];
	}
	#else
	br[i] += ar[j];
	if (com){
		bi[i] += ai[j];
	}
	#endif
}
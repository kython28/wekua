#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void gemm_2(
	__global wk *ar, __global wk *ai,
	__global wks *br, __global wks *bi,

	unsigned long col, unsigned long col2,
	unsigned char com
){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	__global wk *arow = &ar[i*col];

	wk ra = 0;

	for (unsigned long k=0; k<col; k++){
		ra += arow[k];
	}

	#if width == 1
	br[i*col2 + j] = ra;
	#else
	br[i*col2 + j] = dot(ra, (wk)(1));
	#endif
}
#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void sum_kernel(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	unsigned long col, unsigned char com
){
	unsigned long i = get_global_id(0);
	unsigned long k = i*col;

	wk ra = ar[k], rb;
	if (com){
		rb = ai[k];
		for (unsigned long j=1; j<col; j++){
			ra += ar[k + j];
			rb += ai[k + j];
		}

		#if width == 1
		bi[i] = rb;
		#else
		bi[i] = sum(rb);
		#endif
	}else{
		for (unsigned long j=1; j<col; j++){
			ra += ar[k + j];
		}
	}

	#if width == 1
	br[i] = ra;
	#else
	br[i] = sum(ra);
	#endif
}
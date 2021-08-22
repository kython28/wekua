#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void sum_kernel(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	unsigned long col
){
	unsigned long i = get_global_id(0);
	unsigned long j = i*col;
	col += j;

	wk ra = ar[j];
#if com
	wk rb = ai[j];
	j++;
	for (; j<col; j++){
		ra += ar[j];
		rb += ai[j];
	}

	#if width == 1
	bi[i] = rb;
	#else
	bi[i] = sum(rb);
	#endif
#else
	j++;
	for (; j<col; j++){
		ra += ar[j];
	}
#endif

	#if width == 1
	br[i] = ra;
	#else
	br[i] = sum(ra);
	#endif
}
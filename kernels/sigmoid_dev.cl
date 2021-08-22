#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void sigmoid_dev(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi){
	unsigned long i = get_global_id(0);
	
	wk aa;
#if com
	wk bb, cc, dd;
	aa = ar[i];
	bb = ai[i];

	cc = 1 - aa;
	dd = -bb;
	complex_mul(&aa, &bb, cc, dd);

	br[i] = aa;
	bi[i] = bb;
#else
	aa = ar[i];
	br[i] = (1 - aa)*aa;
#endif
}
#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void tanh_dev(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	unsigned char com){
	unsigned long i = get_global_id(0);
	
	wk aa, bb;
	if (com){
		aa = ar[i];
		bb = ai[i];
		complex_mul(&aa, &bb, aa, bb);

		aa = 1 - aa;
		bb *= -1;

		br[i] = aa;
		bi[i] = bb;
	}else{
		aa = ar[i];
		br[i] = 1 - aa*aa;
	}
}
#include "/usr/lib/wekua_kernels/dtype.cl"

// *************************************************************************
// This is not finished yet, come back soon (Only works for arrays with real elements)
// *************************************************************************

__kernel void gauss2(
	__global wks *ar, __global wks *ai,
	__global wks *br, __global wks *bi,
	unsigned long col, unsigned char com
){
	unsigned long i = get_global_id(0);
	unsigned long j = i*col + get_global_id(1);
	i += i*col;

	if (com){
		wks aa, bb;
		aa = br[i]; bb = bi[i];

		calc_inv_complex_scal(&aa, &bb);
		complex_mul_scal2(&aa, &bb, ar[j], ai[j]);

		ar[j] = aa;
		ai[j] = bb;

	}else{
		ar[j] /= br[i];
	}
}
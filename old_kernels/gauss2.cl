#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void gauss2(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	unsigned long col
){
	unsigned long i = get_global_id(0);
	unsigned long j = i*col + get_global_id(1);

	#if wk_width == 1
	i += i*col;
	#else
	unsigned long mod = i%wk_width;

	i = i*col + (i - mod)/wk_width;
	#endif

#if com
	wk aa, bb;
	wks cc, dd;
	aa = ar[j]; bb = ai[j];

	#if wk_width == 1
	cc = br[i];
	dd = bi[i];
	#else
	cc = br[i][mod];
	dd = bi[i][mod];
	#endif

	calc_inv_complex_scal(&cc, &dd);
	complex_mul_scal(&aa, &bb, cc, dd);

	ar[j] = aa;
	ai[j] = bb;
#else
	#if wk_width == 1
	ar[j] /= br[i];
	#else
	ar[j] /= br[i][mod];
	#endif
#endif
}
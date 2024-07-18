#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void elu(
	__global wks *ar, __global wks *ai,
	unsigned long col,
	wks alpha, wks beta
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wks aa, bb, cc, dd;

	aa = ar[i];

	if (aa < 0.0){
#if com
		bb = ai[i];
		aa = exp(aa);

		cc = aa*cos(bb) - 1.0;
		dd = aa*sin(bb);

		complex_mul_scal(&cc, &dd, alpha, beta);

		ar[i] = cc;
		ai[i] = dd;
#else
		ar[i] = alpha*(exp(aa) - 1);
#endif
	}
}
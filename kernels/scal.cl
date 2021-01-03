#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void scal(__global wk *a, __global wk *b,
	wks alpha, wks beta, unsigned char com){
	unsigned long i = get_global_id(0);

	wk aa, bb;
	if (com){
		aa = a[i]; bb = b[i];

		#if width == 1
		complex_mul(&aa, &bb, alpha, beta);
		#else
		complex_mul(&aa, &bb, (wk)(alpha), (wk)(beta));
		#endif
		a[i] = aa;
		b[i] = bb;
	}else{
		a[i] *= alpha;
	}
}
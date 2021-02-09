#include "/usr/lib/wekua_kernels/dtype.cl"

#if dtype == 8
#define ONE 1.0f
#else
#define ONE 1.0
#endif

__kernel void sigmoid(__global wk *a, __global wk *b,
	unsigned char com){

	unsigned long i = get_global_id(0);

	wk aa, bb, c , d;
	c = exp(-ONE*a[i]);
	
	if (com){
		bb = b[i];

		#if width == 1
		d = ONE + c*cos(bb);
		#else
		d = (wk)(ONE) + c*cos(bb);
		#endif
		aa = -ONE*c*sin(bb);

		calc_inv_complex(&d, &aa);

		a[i] = d;
		b[i] = aa;
	}else{
		#if width == 1
		a[i] = ONE/(c+ONE);
		#else
		a[i] = (wk)(ONE)/(c+ (wk)(ONE));
		#endif
	}
}
#include "/usr/lib/wekua_kernels/dtype.cl"

void calc_inv_complex(wk *a, wk *b){
	wk c, d, aa, bb;
	aa = a[0]; bb = b[0];

	c = aa;
	d = -1.0*bb;

	aa = (aa*aa + bb*bb);

	c /= aa;
	d /= aa;
	
	a[0] = c;
	b[0] = d;
}
__kernel void sigmoid(__global wk *a, __global wk *b,
	unsigned char com){

	unsigned long i = get_global_id(0);

	wk aa, bb, c , d;
	c = exp(-1.0*a[i]);
	
	if (com){
		bb = b[i];

		#if width == 1
		d = 1.0 + c*cos(bb);
		#else
		d = (wk)(1.0) + c*cos(bb);
		#endif
		aa = -1.0*c*sin(bb);

		calc_inv_complex(&d, &aa);

		a[i] = d;
		b[i] = aa;
	}else{
		#if width == 1
		a[i] = 1.0/(c+1.0);
		#else
		a[i] = (wk)(1.0)/(c+ (wk)(1.0));
		#endif
	}
}
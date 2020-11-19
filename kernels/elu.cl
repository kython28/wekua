#include "/usr/lib/wekua_kernels/dtype.cl"

void complex_mul(wks *a, wks *b, wks c, wks d){
	wks e, f, g, h;
	g = a[0]; h = b[0];
	e = g*c - h*d;
	f = g*d + h*c;
	a[0] = e;
	b[0] = f;
}

__kernel void elu(
	__global wks *ar, __global wks *ai,
	unsigned long col,
	wks alpha, wks beta, unsigned char com
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wks aa, bb, cc, dd;

	aa = ar[i];

	if (aa < 0.0){
		if (com){
			bb = ai[i];
			aa = exp(aa);

			cc = aa*cos(bb) - 1.0;
			dd = aa*sin(bb);

			complex_mul(&cc, &dd, alpha, beta);

			ar[i] = cc;
			ai[i] = dd;
		}else{
			ar[i] = alpha*(exp(ar[i]) - 1);
		}
	}
}
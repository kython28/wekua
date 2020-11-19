#include "/usr/lib/wekua_kernels/dtype.cl"

void complex_mul(wks *a, wks *b, wkss c, wkss d){
	wks e, f, g, h;
	g = a[0]; h = b[0];
	e = g*c - h*d;
	f = g*d + h*c;
	a[0] = e;
	b[0] = f;
}

__kernel void leakyrelu(
	__global wks *ar, __global wks *ai,
	unsigned long col,
	wkss alpha, wkss beta, unsigned char com
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);
	
	wks aa, bb;

	aa = ar[i];
	if (com){
		bb = ai[i];
		complex_mul(&aa, &bb, alpha, beta);

		if (aa > ar[i] && bb > ai[i]){
			ar[i] = aa;
			ai[i] = bb;
		}
	}else{
		aa *= alpha;

		if (aa > ar[i]){
			ar[i] = aa;
		}
	}
}
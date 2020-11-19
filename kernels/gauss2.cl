#include "/usr/lib/wekua_kernels/dtype.cl"

void calc_inv_complex(wks *a, wks *b){
	wks c, d, aa, bb;
	aa = a[0]; bb = b[0];

	c = aa;
	d = -1.0*bb;

	aa = (aa*aa + bb*bb);

	c /= aa;
	d /= aa;
	
	a[0] = c;
	b[0] = d;
}

void complex_mul(wks *a, wks *b, wks c, wks d){
	wks e, f, g, h;
	g = a[0]; h = b[0];
	e = g*c - h*d;
	f = g*d + h*c;
	a[0] = e;
	b[0] = f;
}

__kernel void gauss2(
	__global wks *ar, __global wks *ai,
	__global wks *br, __global wks *bi,
	unsigned long col, unsigned char com,
	__local wks *atiler, __local wks *atilei
){
	unsigned long i = get_global_id(0);
	unsigned long j = i*col + get_global_id(1);

	unsigned long x = get_local_id(0);
	unsigned long y = get_local_id(1);

	if (y == 0){
		atiler[x] = br[i*col+i];
		if (com){
			atilei[x] = bi[i*col+i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (com){
		wks aa, bb;
		aa = atiler[x]; bb = atilei[x];

		calc_inv_complex(&aa, &bb);
		complex_mul(&aa, &bb, ar[j], ai[j]);

		ar[j] = aa;
		ai[j] = bb;

	}else{
		ar[j] /= atiler[x];
	}


}
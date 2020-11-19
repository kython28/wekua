#include "/usr/lib/wekua_kernels/dtype.cl"

void calc_inv_complex(wks *a, wks *b){
	wks c, d, aa, bb;
	aa = a[0]; bb = b[0];

	c = aa;
	d = -1.0*bb;

	aa = (aa*aa + bb*bb);

	c /= aa;
	d /= bb;
	
	a[0] = c;
	b[0] = d;
}

void gauss_real(__global wks *a, __global wks *b, __global wks *c, unsigned long k, unsigned long col){
	wks a_c;
	a_c = b[k]/a[k];

	c[0] /= a_c;
	for (unsigned long x=k; x<col; x++){
		a[x] = a[x]*a_c - b[x];
	}
}

void gauss_complex(__global wks *ar, __global wks *ai, __global wks *br, __global wks *bi, __global wks *cr,
	__global wks *ci, unsigned long k, unsigned long col){
	wks aa, bb, cc, dd, ee, ff;

	ee = br[k]; ff = bi[k];
	cc = ar[k]; dd = ai[k];
	calc_inv_complex(&cc, &dd);

	aa = ee*cc - ff*dd;
	bb = ee*dd + ff*cc;

	ee = aa; ff = bb;
	calc_inv_complex(&ee, &ff);

	cr[0] = ee;
	ci[0] = ff;

	for (unsigned long x=k; x<col; x++){
		ee = ar[x]; ff = ai[x];

		ar[x] = ee*aa - ff*bb - br[x];
		ai[x] = ee*bb + ff*aa - bi[x];
	}
}


__kernel void det(
	__global wks *ar, __global wks *ai,
	__global wks *br, __global wks *bi,
	unsigned long k, unsigned long col,
	unsigned long rcol, unsigned char com
){
	unsigned long i = get_global_id(0);

	if (i > k){
		if (com){
			gauss_complex(
				&ar[i*rcol], &ai[i*rcol], &ar[k*rcol], &ai[k*rcol],
				&br[i*rcol], &bi[i*rcol], k, col
			);
		}else{
			gauss_real(&ar[i*rcol], &ar[k*rcol], &br[i*rcol], k, col);
		}
	}else if (i == k){
		br[k*rcol + k] = ar[k*rcol + k];
		if (com) bi[k*rcol + k] = ai[k*rcol + k];
	}
}
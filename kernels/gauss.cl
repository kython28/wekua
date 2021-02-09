#include "/usr/lib/wekua_kernels/dtype.cl"

void gauss_real(__global wks *a, __global wks *b, __global wks *c, __global wks *d, unsigned long k, unsigned char otherm, unsigned long col){
	wks a_c, b_c;
	a_c = b[k]/a[k];

	for (unsigned long x=0; x<col; x++){
		a[x] = a[x]*a_c - b[x];
		if (otherm){
			c[x] = c[x]*a_c - d[x];
		}
	}
}

void gauss_complex(__global wks *ar, __global wks *ai, __global wks *br, __global wks *bi, __global wks *cr,
	__global wks *ci, __global wks *dr, __global wks *di, unsigned long k, unsigned char otherm, unsigned long col){
	wks aa, bb, cc, dd, ee, ff;

	ee = br[k]; ff = bi[k];
	cc = ar[k]; dd = ai[k];
	calc_inv_complex_scal(&cc, &dd);

	aa = ee*cc - ff*dd;
	bb = ee*dd + ff*cc;

	for (unsigned long x=0; x<col; x++){
		ee = ar[x]; ff = ai[x];

		ar[x] = ee*aa - ff*bb - br[x];
		ai[x] = ee*bb + ff*aa - bi[x];

		if (otherm){
			ee = cr[x]; ff = ci[x];

			cr[x] = ee*aa - ff*bb - dr[x];
			ci[x] = ee*bb + ff*aa - di[x];
		}
	}
}


__kernel void gauss(
	__global wks *ar, __global wks *ai,
	__global wks *br, __global wks *bi,
	unsigned long k, unsigned long col,
	unsigned long rcol, unsigned char otherm,
	unsigned char up, unsigned char com
){
	unsigned long i = get_global_id(0);

	if (i != k){
		if ((up && i < k) || i > k){
			if (com){
				if (otherm){
					gauss_complex(
						&ar[i*rcol], &ai[i*rcol], &ar[k*rcol], &ai[k*rcol],
						&br[i*rcol], &bi[i*rcol], &br[k*rcol], &bi[k*rcol],
						k, otherm, col
					);
				}else{
					gauss_complex(
						&ar[i*rcol], &ai[i*rcol], &ar[k*rcol], &ai[k*rcol],
						0, 0, 0, 0,
						k, otherm, col
					);
				}
				
			}else{
				if (otherm){
					gauss_real(&ar[i*rcol], &ar[k*rcol], &br[i*rcol], &br[k*rcol], k, otherm, col);
				}else{
					gauss_real(&ar[i*rcol], &ar[k*rcol], 0, 0, k, otherm, col);
				}
			}
		}
	}
}
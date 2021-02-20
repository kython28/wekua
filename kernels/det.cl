#include "/usr/lib/wekua_kernels/dtype.cl"

// *************************************************************************
// This is not finished yet, come back soon
// *************************************************************************

void gauss_real(__global wk *a, __global wk *b, __global wk *c, unsigned long k, unsigned long col){
	wks a_c;

	#if width == 1
	a_c = b[k]/a[k];
	#else
	unsigned long mod = k%width, addr;
	addr = (k - mod)/width;

	a_c = b[addr][mod]/a[addr][mod];
	#endif

	#if width == 1
	c[0] /= a_c;
	#else
	c[0][0] /= a_c;
	#endif

	for (unsigned long x=0; x<col; x++){
		a[x] = a[x]*a_c - b[x];
	}
}


void gauss_complex(__global wk *ar, __global wk *ai, __global wk *br, __global wk *bi, __global wk *cr,
	__global wk *ci, unsigned long k, unsigned long col){
	wks aa, bb, cc, dd;

	wk ee, ff;

	#if width == 1
	aa = br[k]; aa = bi[k];
	cc = ar[k]; dd = ai[k];
	#else
	unsigned long mod = k%width, addr;
	addr = (k - mod)/width;

	aa = br[addr][mod]; bb = bi[addr][mod];
	cc = ar[addr][mod]; dd = ai[addr][mod];
	#endif

	calc_inv_complex_scal(&cc, &dd);
	complex_mul_scal2(&aa, &bb, cc, dd);

	for (unsigned long x=0; x<col; x++){
		ee = ar[x]; ff = ai[x];

		complex_mul_scal(&ee, &ff, aa, bb);

		ar[x] = ee - br[x];
		ai[x] = ff - bi[x];

		if (otherm){
			ee = cr[x]; ff = ci[x];

			complex_mul_scal(&ee, &ff, aa, bb);

			cr[x] = ee - dr[x];
			ci[x] = ff - di[x];
		}
	}
}


__kernel void det(
	__global wks *ar, __global wks *ai,
	__global wks *br, __global wks *bi,
	unsigned long k, unsigned long col,
	unsigned char com
){
	unsigned long i = get_global_id(0);

	if (i > k){
		if (com){
			gauss_complex(
				&ar[i*col], &ai[i*col], &ar[k*col], &ai[k*col],
				&br[i*col], &bi[i*col], k, col
			);
		}else{
			gauss_real(&ar[i*col], &ar[k*col], &br[i*col], k, col);
		}
	}else if (i == k){
		br[k*col + k] = ar[k*col + k];
		if (com) bi[k*col + k] = ai[k*col + k];
	}
}
#include "/usr/lib/wekua_kernels/dtype.cl"

#if com
void gauss_complex(__global wk *ar, __global wk *ai, __global wk *br, __global wk *bi, __global wk *cr,
	__global wk *ci, __global wk *dr, __global wk *di, unsigned long k, unsigned char otherm, unsigned long col){
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

	if (isnan(aa) || isnan(bb) || isinf(aa) || isinf(bb)) return;

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
#else
void gauss_real(__global wk *a, __global wk *b, __global wk *c, __global wk *d, unsigned long k, unsigned char otherm, unsigned long col){
	wks a_c;

	#if width == 1
	a_c = b[k]/a[k];
	#else
	unsigned long mod = k%width, addr;
	addr = (k - mod)/width;

	a_c = b[addr][mod]/a[addr][mod];
	#endif

	if (isnan(a_c) || isinf(a_c)) return;

	for (unsigned long x=0; x<col; x++){
		a[x] = a[x]*a_c - b[x];
		if (otherm){
			c[x] = c[x]*a_c - d[x];
		}
	}
}
#endif

__kernel void gauss(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	unsigned long k, unsigned long col,
	unsigned char otherm, unsigned char up
){
	unsigned long i = get_global_id(0);

	if (i != k){
		if ((up && i < k) || i > k){
#if com
			if (otherm){
				gauss_complex(
					&ar[i*col], &ai[i*col], &ar[k*col], &ai[k*col],
					&br[i*col], &bi[i*col], &br[k*col], &bi[k*col],
					k, otherm, col
				);
			}else{
				gauss_complex(
					&ar[i*col], &ai[i*col], &ar[k*col], &ai[k*col],
					0, 0, 0, 0,
					k, otherm, col
				);
			}
#else
			if (otherm){
				gauss_real(&ar[i*col], &ar[k*col], &br[i*col], &br[k*col], k, otherm, col);
			}else{
				gauss_real(&ar[i*col], &ar[k*col], 0, 0, k, otherm, col);
			}
#endif
		}
	}
}
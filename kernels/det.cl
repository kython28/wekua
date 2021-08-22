#include "/usr/lib/wekua_kernels/dtype.cl"

#if com
void gauss(__global wk *ar, __global wk *ai, __global wk *br, __global wk *bi, __global wk *cr,
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

	if (isnan(aa) || isnan(bb) || isinf(aa) || isinf(bb)) return;

	cc = aa; dd = bb;
	calc_inv_complex_scal(&cc, &dd);

	#if width == 1
	complex_mul_scal2(&cc, &dd, cr[k], ci[k]);

	cr[k] = cc;
	ci[k] = dd;
	for (unsigned long x=0; x<col; x++)
	#else
	complex_mul_scal2(&cc, &dd, cr[addr][mod], ci[addr][mod]);

	cr[addr][mod] = cc;
	ci[addr][mod] = dd;

	for (unsigned long x=addr; x<col; x++)
	#endif
	{
		ee = ar[x]; ff = ai[x];
		complex_mul_scal(&ee, &ff, aa, bb);

		ar[x] = ee - br[x];
		ai[x] = ff - bi[x];
	}
}
#else
void gauss(__global wk *a, __global wk *b, __global wk *c, unsigned long k, unsigned long col){
	wks a_c;

	#if width == 1
	a_c = b[k]/a[k];

	if (isnan(a_c) || isinf(a_c)) return;

	c[k] /= a_c;

	for (unsigned long x=k; x<col; x++)
	#else
	unsigned long mod = k%width, addr;
	addr = (k - mod)/width;

	a_c = b[addr][mod]/a[addr][mod];

	if (isnan(a_c) || isinf(a_c)) return;

	c[addr][mod] /= a_c;
	for (unsigned long x=addr; x<col; x++)
	#endif
	{
		a[x] = a[x]*a_c - b[x];
	}
}
#endif


__kernel void det(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	unsigned long k, unsigned long col,
	unsigned char com
){
	unsigned long i = get_global_id(0);

	if (i > k){
#if com
		gauss(
			&ar[i*col], &ai[i*col], &ar[k*col], &ai[k*col],
			&br[i*col], &bi[i*col], k, col
		);
#else
		gauss(&ar[i*col], &ar[k*col], &br[i*col], k, col);
#endif
	}else if (i == k){
#if width == 1
		k = k*col + k;
		br[k] = ar[k];
#if com
		bi[k] = ai[k];
#endif
#else
		unsigned long mod = k%width, addr;
		addr = (k - mod)/width;

		k *= col;
		k += addr;

		br[k][mod] = ar[k][mod];
#if com
		bi[k][mod] = ai[k][mod];
#endif
#endif
	}
}
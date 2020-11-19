#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void gemm(
	__global wk *ar, __global wk *ai,
	__global wk *br, __global wk *bi,
	__global wks *cr, __global wks *ci,
	wks ralpha, wks ialpha,
	wks rbeta, wks ibeta,
	unsigned long col, unsigned long col2,
	unsigned long col3, unsigned char com
){
	unsigned long i = get_global_id(0);
	unsigned long k = get_global_id(1);

	unsigned long ccurr = i*col3 + k;
	unsigned long arow = i*col;
	unsigned long brow = k*col2;

	wks ra = 0, rb = 0, crr, cri;
	wk aa, bb, cc, dd;

	if (com){
		for (unsigned long j=0; j<col; j++){
			aa = ar[arow+j]; bb = ai[arow+j];
			cc = br[brow+j]; dd = bi[brow+j];
		#if width == 1
			ra += aa*cc - bb*dd;
			rb += aa*dd + bb*cc;
		#else
			ra += dot(aa*cc - bb*dd, (wk)(1));
			rb += dot(aa*dd + bb*cc, (wk)(1));
		#endif
		}
		crr = cr[ccurr]; cri = ci[ccurr];

		cr[ccurr] = ra*ralpha - rb*ialpha + crr*rbeta - cri*ibeta;
		ci[ccurr] = ra*ialpha + rb*ralpha + crr*ibeta + cri*rbeta;
	}else{
		for (unsigned long j=0; j<col; j++){
		#if width == 1
			ra += ar[arow+j]*br[brow+j];
		#else
			ra += dot(ar[arow+j], br[brow+j]);
		#endif
		}
		cr[ccurr] = ra*ralpha + rbeta*cr[ccurr];
	}
}
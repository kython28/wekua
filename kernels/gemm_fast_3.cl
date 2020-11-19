#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void gemm_fast_3(
	__global wks *ar, __global wks *ai,
	__global wks *br, __global wks *bi,
	__global wks *cr, __global wks *ci,

	unsigned long c, unsigned long c2,
	unsigned long c3, unsigned long m,
	unsigned char com
){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	unsigned long x = get_local_id(0);
	unsigned long y = get_local_id(1);

	unsigned long cur = i*c3 + j, rl, rl2, rl3;

	#if ks == 1 || ks == 2 || ks == 3 || ks == 4
	unsigned long rl4;
	#endif


	#if ks == 2
	rl = (i << 1)*c;
	#elif ks == 3
	rl = ((i << 1) + 1)*c;
	#else
	rl = i*c;
	#endif

	#if ks == 1
	rl2 = (j << 1)*c2;
	#elif ks == 4
	rl2 = ((j << 1) + 1)*c2;
	#else
	rl2 = j*c2;
	#endif

	wks ra = 0, rb = 0;

	wks bb, cc;
	wks dd, ee;

	wks bbi, cci, ddi, eei;

	if (com){
		for (unsigned long k=0; k<m; k++){
			#if ks == 1
			rl3 = (k << 1) + 1; rl4 = (k << 2) + 3;

			bb = ar[rl + rl3 - 1]; cc = ar[rl + rl3];
			dd = br[rl2 + rl4 - 1]; ee = br[rl2 + rl4 - 3];

			bbi = ai[rl + rl3 - 1]; cci = ai[rl + rl3];
			ddi = bi[rl2 + rl4 - 1]; eei = bi[rl2 + rl4 - 3];
			#elif ks == 4
			rl3 = (k << 1) + 1; rl4 = (k << 2) + 3;

			bb = ar[rl + rl3 - 1]; cc = ar[rl + rl3];
			dd = br[rl2 + rl4]; ee = br[rl2 + rl4 - 2];

			bbi = ai[rl + rl3 - 1]; cci = ai[rl + rl3];
			ddi = bi[rl2 + rl4]; eei = bi[rl2 + rl4 - 2];
			#elif ks == 2
			rl3 = (k << 2) + 3; rl4 = (k << 1) + 1;

			bb = ar[rl + rl3 - 3]; cc = ar[rl + rl3 - 1];
			dd = br[rl2 + rl4]; ee = br[rl2 + rl4 - 1];

			bbi = ai[rl + rl3 - 3]; cci = ai[rl + rl3 - 1];
			ddi = bi[rl2 + rl4]; eei = bi[rl2 + rl4 - 1];
			#elif ks == 3
			rl3 = (k << 2) + 3; rl4 = (k << 1) + 1;

			bb = ar[rl + rl3 - 2]; cc = ar[rl + rl3];
			dd = br[rl2 + rl4]; ee = br[rl2 + rl4 - 1];

			bbi = ai[rl + rl3 - 2]; cci = ai[rl + rl3];
			ddi = bi[rl2 + rl4]; eei = bi[rl2 + rl4 - 1];
			#else
			rl3 = (k << 1) + 1;

			bb = ar[rl + rl3 - 1]; cc = ar[rl + rl3];
			dd = br[rl2 + rl3]; ee = br[rl2 + rl3 - 1];

			bbi = ai[rl + rl3 - 1]; cci = ai[rl + rl3];
			ddi = bi[rl2 + rl3]; eei = bi[rl2 + rl3 - 1];
			#endif

			ra += bb*ee + cc*dd - bbi*eei - cci*ddi;
			rb += bb*eei + bbi*ee + cc*ddi + cci*dd;
		}
		ci[cur] = rb;
	}else{
		for (unsigned long k=0; k<m; k++){
			#if ks == 1
			rl3 = (k << 1) + 1; rl4 = (k << 2) + 3;

			bb = ar[rl + rl3 - 1]; cc = ar[rl + rl3];
			dd = br[rl2 + rl4 - 1]; ee = br[rl2 + rl4 - 3];
			#elif ks == 4
			rl3 = (k << 1) + 1; rl4 = (k << 2) + 3;

			bb = ar[rl + rl3 - 1]; cc = ar[rl + rl3];
			dd = br[rl2 + rl4]; ee = br[rl2 + rl4 - 2];
			#elif ks == 2
			rl3 = (k << 2) + 3; rl4 = (k << 1) + 1;

			bb = ar[rl + rl3 - 3]; cc = ar[rl + rl3 - 1];
			dd = br[rl2 + rl4]; ee = br[rl2 + rl4 - 1];
			#elif ks == 3
			rl3 = (k << 2) + 3; rl4 = (k << 1) + 1;

			bb = ar[rl + rl3 - 2]; cc = ar[rl + rl3];
			dd = br[rl2 + rl4]; ee = br[rl2 + rl4 - 1];
			#else
			rl3 = (k << 1) + 1;

			bb = ar[rl + rl3 - 1]; cc = ar[rl + rl3];
			dd = br[rl2 + rl3]; ee = br[rl2 + rl3 - 1];
			#endif

			ra += bb*ee + cc*dd;
		}
	}
	cr[cur] = ra;
}
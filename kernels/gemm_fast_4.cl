#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void gemm_fast_4(
	__global wks *ar1, __global wks *ai1,
	__global wks *ar2, __global wks *ai2,
	__global wks *ar3, __global wks *ai3,
	__global wks *ar4, __global wks *ai4,
	__global wks *ar5, __global wks *ai5,
	__global wks *ar6, __global wks *ai6,
	__global wks *ar7, __global wks *ai7,

	wks ralpha, wks ialpha,
	wks rbeta, wks ibeta,

	__global wks *cr, __global wks *ci,
	unsigned long c, unsigned long c2,
	unsigned char com
){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	unsigned long cur = i*c2 + j;
	unsigned long rl, rl2, rl3, rl4;

	wks aa, bb, cc, dd, ee;
	wks aai, bbi, cci, ddi, eei;
	wks ra, rb, rc, rd;

	rl = (i << 1)*c;
	rl2 = ((i << 1) + 1)*c;
	rl3 = (j << 1);
	rl4 = rl3 + 1;

	aa = ar1[cur]; bb = ar2[cur];
	cc = ar3[cur]; dd = ar4[cur];
	ee = ar5[cur];

	if (com){
		aai = ai1[cur]; bbi = ai2[cur];
		cci = ai3[cur]; ddi = ai4[cur];
		eei = ai5[cur];

		// -------- 1 -------- //
		ra = aa + dd - ee + ar7[cur];
		rb = aai + ddi - eei + ai7[cur];

		rc = cr[rl + rl3];
		rd = ci[rl + rl3];

		cr[rl + rl3] = ralpha*ra - ialpha*rb + rbeta*rc - rd*ibeta;
		ci[rl + rl3] = ralpha*rb + ialpha*ra + rbeta*rd + rc*ibeta;

		// -------- 2 -------- //

		ra = cc + ee;
		rb = cci + eei;

		rc = cr[rl + rl4];
		rd = ci[rl + rl4];

		cr[rl + rl4] = ralpha*ra - ialpha*rb + rbeta*rc - rd*ibeta;
		ci[rl + rl4] = ralpha*rb + ialpha*ra + rbeta*rd + rc*ibeta;

		// -------- 3 -------- //

		ra = bb + dd;
		rb = bbi + ddi;

		rc = cr[rl2 + rl3];
		rd = ci[rl2 + rl3];

		cr[rl2 + rl3] = ralpha*ra - ialpha*rb + rbeta*rc - rd*ibeta;
		ci[rl2 + rl3] = ralpha*rb + ialpha*ra + rbeta*rd + rc*ibeta;

		// -------- 4 -------- //

		ra = aa - bb + cc + ar6[cur];
		rb = aai - bbi + cci+ ai6[cur];

		rc = cr[rl2 + rl4];
		rd = ci[rl2 + rl4];

		cr[rl2 + rl4] = ralpha*ra - ialpha*rb + rbeta*rc - rd*ibeta;
		ci[rl2 + rl4] = ralpha*rb + ialpha*ra + rbeta*rd + rc*ibeta;
	}else{
		cr[rl + rl3] = ralpha*(aa + dd - ee + ar7[cur]) + rbeta*cr[rl + rl3];
		cr[rl + rl4] = ralpha*(cc + ee) + rbeta*cr[rl + rl4];
		cr[rl2 + rl3] = ralpha*(bb + dd) + rbeta*cr[rl2 + rl3];
		cr[rl2 + rl4] = ralpha*(aa - bb + cc + ar6[cur]) + rbeta*cr[rl2 + rl4];
	}
}
#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void gemm_fast_2(
	__global wks *ar, __global wks *ai,

	__global wks *alr1, __global wks *ali1,
	__global wks *alr2, __global wks *ali2,
	__global wks *alr3, __global wks *ali3,
	__global wks *alr4, __global wks *ali4,
	__global wks *alr5, __global wks *ali5,

	unsigned long c, unsigned long c2,
	unsigned char com
){
	unsigned long i = get_global_id(0);
	unsigned long k = get_global_id(1);

	unsigned long rl, rl2, rl3, rl4;
	rl = (i << 1)*c; rl2 = ((i << 1) + 1)*c;
	rl3 = (k << 1); rl4 = i*c2 + k;

	wks aa, bb, cc, dd;

	aa = ar[rl + rl3]; // 2k - 1, 2j - 1
	bb = ar[rl + rl3 + 1]; // 2k, 2j - 1
	cc = ar[rl2 + rl3]; // 2k - 1, 2j
	dd = ar[rl2 + rl3 + 1]; // 2k, 2j

	alr1[rl4] = aa + dd;
	alr2[rl4] = cc - dd;
	alr3[rl4] = bb - aa;
	alr4[rl4] = aa + cc;
	alr5[rl4] = bb + dd;

	if (com){
		aa = ai[rl + rl3];
		bb = ai[rl + rl3 + 1];
		cc = ai[rl2 + rl3];
		dd = ai[rl2 + rl3 + 1];

		ali1[rl4] = aa + dd;
		ali2[rl4] = cc - dd;
		ali3[rl4] = bb - aa;
		ali4[rl4] = aa + cc;
		ali5[rl4] = bb + dd;
	}
}
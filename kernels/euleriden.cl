#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void euler_iden(
	__global wks *ar, __global wks *ai,
	__global wks *br, __global wks *bi,
	unsigned long col, unsigned char com
){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

	wks aa;
	if (com){
		wks bb, cc;
		aa = ar[i]; bb = ai[i];

		cc = cosh(bb) - sinh(bb);

		br[i] = cos(aa)*cc;
		bi[i] = sin(aa)*cc;
	}else{
		aa = ar[i];
		br[i] = cos(aa);
		bi[i] = sin(aa);
	}
}
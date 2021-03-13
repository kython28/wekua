#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void leakyrelu(
	__global wks *ar, __global wks *ai,
	wks alpha, wks alphai,
	unsigned long col, unsigned char com
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	if (ar[i] < 0.0){
		if (com){
			wks aa, bb;
			aa = ar[i];
			bb = ai[i];

			complex_mul_scal2(&aa, &bb, alpha, alphai);

			ar[i] = aa;
			ai[i] = bb;	
		}else{
			ar[i] *= alpha;	
		}
	}
}
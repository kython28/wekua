#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void scal(__global wk *a, __global wk *b,
	wks alpha, wks beta, unsigned char com){
	unsigned long i = get_global_id(0);

	wk aa, bb;
	if (com){
		aa = a[i]; bb = b[i];
		a[i] = aa*alpha - bb*beta;
		b[i] = aa*beta + bb*alpha;
	}else{
		a[i] *= alpha;
	}
}
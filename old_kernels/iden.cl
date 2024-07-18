#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void iden(__global wks *a, unsigned long col){
	unsigned long i = get_global_id(0);

	a[i*col+i] = 1;
}
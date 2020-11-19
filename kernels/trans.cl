#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void trans(__global wks *a, __global wks *b,
	__global wks *c, __global wks *d,
	unsigned long col, unsigned long row,
	unsigned char com){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	c[j*row+i] = a[i*col+j];
	if (com){
		d[j*row+i] = b[i*col+j];
	}
}
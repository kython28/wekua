#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void diag(__global wks *a, __global wks *b,
	__global wks *c, __global wks *d,
	unsigned long col, unsigned char mode, unsigned char com){
	unsigned long i = get_global_id(0);

	if (mode){ // To get diag from square Matrix
		c[i] = a[i*col+i];
		if (com){
			d[i] = b[i*col+i];
		}
	}else{ // Diag to Square Matrix
		c[i*col+i] = a[i];
		if (com){
			d[i*col+i] = b[i];
		}
	}

}
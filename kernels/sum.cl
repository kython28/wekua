__kernel void sum(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	c[i] += a[i*col+j];
	if (com){
		d[i] += b[i*col+j];
	}

}
__kernel void sum(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned char com){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);

	c[i] += a[i*col+j];
	if (com){
		d[i] += b[i*col+j];
	}

}
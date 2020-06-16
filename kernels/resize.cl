__kernel void resize(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned int col2, unsigned char com){
	
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);

	a[i*col+j] = c[i*col2+j];
	if (com){
		b[i*col+j] = d[i*col2+j];
	}

}
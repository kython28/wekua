__kernel void trans(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned int row, unsigned char com){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	a[j*row+i] = c[i*col+j];
	if (com){
		b[j*row+i] = d[i*col+j];
	}
}
__kernel void axpy(__global double *a, __global double *b,
	__global double *c, __global double *d, unsigned int col,
	unsigned char com, double alpha){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	a[i*col+j] += alpha*c[i*col+j];
	if (com){
		b[i*col+j] += alpha*d[i*col+j];
	}
}
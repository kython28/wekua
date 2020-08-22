__kernel void axpy(__global double *a, __global double *b,
	__global double *c, __global double *d, unsigned long col,
	unsigned char com, double alpha){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);
	a[i*col+j] += alpha*c[i*col+j];
	if (com){
		b[i*col+j] += alpha*d[i*col+j];
	}
}
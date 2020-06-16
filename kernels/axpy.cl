__kernel void axpy(__global double *a, __global double *b,
	__global double *c, __global double *d, unsigned int col,
	unsigned char com, double alpha){
	unsigned long i = get_global_id(0);
	a[i] += alpha*c[i];
	if (com){
		b[i] += alpha*d[i];
	}
}
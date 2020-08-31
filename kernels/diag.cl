__kernel void diag(__global double *a, __global double *b,
	__global double *c, __global double *d, unsigned char com){
	unsigned long i = get_global_id(0);
	unsigned long col = get_global_size(0);
	a[i*col+i] = c[i];
	if (com){
		b[i*col+i] = d[i];
	}
}
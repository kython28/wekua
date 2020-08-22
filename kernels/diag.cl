__kernel void diag(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0);
	a[i] = c[i*col+i];
	if (com){
		b[i] = d[i*col+i];
	}
}
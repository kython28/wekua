__kernel void resize(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long col, unsigned long col2,
	unsigned char com){
	
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	a[i*col+j] = c[i*col2+j];
	if (com){
		b[i*col+j] = d[i*col2+j];
	}
}
__kernel void cut(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long col, unsigned long x, unsigned long y,
	unsigned long w, unsigned char com){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);
	a[i*col+j] = c[(y+i)*w+x+j];
	if (com){
		b[i*col+j] = d[(y+i)*w+x+j];	
	}
}
__kernel void trans(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long col, unsigned long row, unsigned char com,
	unsigned long offsetar, unsigned long offsetac){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);
	a[j*row+i] = c[(i+offsetar)*col+j+offsetac];
	if (com){
		b[j*row+i] = d[(i+offsetar)*col+j+offsetac];
	}
}
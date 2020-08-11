__kernel void rand(__global double *a, __global double *b,
	__global long *c, __global long *d, unsigned char com){
	unsigned long i = get_global_id(0);
	a[i] = fabs((c[i] >> 11)*DBL_EPSILON);
	if (com){
		b[i] = fabs((d[i] >> 11)*DBL_EPSILON);
	}
}
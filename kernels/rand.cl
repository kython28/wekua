__kernel void rand(__global double *a, __global double *b,
	__global long *c, __global long *d,
	unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	a[i] = fabs((c[i] >> 11)*DBL_EPSILON);
#if com
	b[i] = fabs((d[i] >> 11)*DBL_EPSILON);
#endif
}
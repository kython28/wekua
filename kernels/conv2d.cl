__kernel void conv2d(
	__global double *a, __global double *b,
	__global double *c, __global double *d,
	__global double *k, __global double *ik,
	unsigned long k1, unsigned long k2,
	unsigned long col, unsigned long col2,
	 
	){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);
	unsigned long k = get_global_id(2);

	double ra = k[klen], rb, aa, bb, cc, dd;
}
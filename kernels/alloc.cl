__kernel void alloc(__global double *a, __global double *b, double alpha, double beta){
	unsigned long i = get_global_id(0);
	a[i] = alpha;
	if (beta != 0.0){
		b[i] = beta;
	}
}
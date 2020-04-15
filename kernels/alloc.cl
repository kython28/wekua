__kernel void alloc(__global double *a, const double alpha){
	unsigned int i = get_global_id(0);
	a[i] = alpha;
}

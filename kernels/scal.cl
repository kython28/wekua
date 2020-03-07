__kernel void scal(__global double *a, double alpha){
	unsigned int i = get_global_id(0);
	a[i] *= alpha;
}
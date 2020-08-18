__kernel void arange(__global double *a, double alpha){
	unsigned long i = get_global_id(0);
	a[i] += alpha*i;
}
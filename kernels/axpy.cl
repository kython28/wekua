__kernel void axpy(__global double *x, __global double *y, const double alpha){
	unsigned int i = get_global_id(0);
	y[i] = alpha*x[i] + y[i];
}

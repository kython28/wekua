__kernel void gaus(__global double *a, __global double *b, unsigned int c){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	b[i*c+j] *= 1/a[i*c+i];
}

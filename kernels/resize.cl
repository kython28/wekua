__kernel void resize(__global double *a, __global double *b, unsigned int c, unsigned int d){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	b[i*d+j] = a[i*c+j];
}
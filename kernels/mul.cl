__kernel void mul(__global double *a, __global double *b, unsigned int c){
	unsigned int i = get_global_id(0);
	unsigned int x = get_global_id(1);
	b[i] *= a[i*c+x];
}
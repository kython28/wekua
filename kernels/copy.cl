__kernel void copy(__global double *a, __global double *b){
	unsigned int i = get_global_id(0);
	a[i] = b[i];
}

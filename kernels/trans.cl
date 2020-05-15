__kernel void trans(__global double *a, __global double *b, unsigned int r, unsigned int c){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	a[j*r+i] = b[i*c+j];
}

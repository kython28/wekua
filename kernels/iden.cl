__kernel void iden(__global double *a, unsigned int c){
	unsigned int i = get_global_id(0);
	a[i*c+i] = 1.0;
}
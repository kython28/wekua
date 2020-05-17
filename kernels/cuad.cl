__kernel void cuad(__global double *a){
	unsigned int i = get_global_id(0);
	a[i] *= a[i];
}
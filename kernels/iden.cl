__kernel void iden(__global double *a, unsigned long c){
	unsigned long i = get_global_id(0);
	a[i*c+i] = 1.0;
}
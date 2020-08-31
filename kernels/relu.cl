__kernel void relu(__global double *a, __global double *b,
	unsigned long col, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

	if (a[i] < 0.0){
		a[i] = 0.0;
		if (com){
			b[i] = 0.0;
		}
	}
}
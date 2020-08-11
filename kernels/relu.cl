__kernel void relu(__global double *a, __global double *b,
	unsigned char com){
	unsigned long i = get_global_id(0);

	if (a[i] < 0.0){
		a[i] = 0.0;
		if (com){
			b[i] = 0.0;
		}
	}
}
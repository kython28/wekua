__kernel void lerelu(__global double *a, __global double *b,
	unsigned char com){
	unsigned long i = get_global_id(0);

	if (a[i] < 0.1*a[i]){
		a[i] = 0.1*a[i];
		if (com){
			b[i] = 0.1*b[i];
		}
	}
}
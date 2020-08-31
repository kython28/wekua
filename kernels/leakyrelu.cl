__kernel void lerelu(__global double *a, __global double *b,
	unsigned long col, double alpha, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

	double aa = a[i];
	if (aa < alpha*aa){
		a[i] = alpha*aa;
		if (com){
			b[i] = alpha*b[i];
		}
	}
}
__kernel void dots(__global double *a, __global double *b, unsigned char com,
	double alpha, double beta){
	unsigned long i = get_global_id(0);
	double aa, bb;
	if (com){
		aa = a[i];
		bb = b[i];
		a[i] = alpha*aa-beta*bb;
		b[i] = alpha*bb+beta*aa;
	}else{
		a[i] *= alpha;
	}
}
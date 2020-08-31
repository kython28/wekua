__kernel void dots(__global double *a, __global double *b, unsigned char com,
	double alpha, double beta, unsigned long col){
	unsigned long current = get_global_id(0)*col+get_global_id(1);
	double aa, bb;
	if (com){
		aa = a[current];
		bb = b[current];
		a[current] = alpha*aa-beta*bb;
		b[current] = alpha*bb+beta*aa;
	}else{
		a[current] *= alpha;
	}
}
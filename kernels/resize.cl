__kernel void resize(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long col, unsigned long col2,
	unsigned long row, unsigned long row2,
	double alpha, double beta, unsigned char com){
	
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	unsigned long curr = i*col+j;
	unsigned long curr2 = i*col2+j;

	if (i < row2 && j < col2){
		a[curr] = c[curr2];
		if (com){
			b[curr] = d[curr2];
		}
	}else{
		a[curr] = alpha;
		if (com){
			b[curr] = beta;
		}
	}
	
}
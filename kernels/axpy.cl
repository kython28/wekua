__kernel void axpy(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned char com, double alpha,
	unsigned long col, unsigned long col2,
	unsigned long offsetar, unsigned long offsetac,
	unsigned long offsetbr, unsigned long offsetbc){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	a[ (i+offsetar)*col + j + offsetac ] += alpha*c[ (i+offsetbr)*col2 + j + offsetbc ];
	if (com){
		b[ (i+offsetar)*col + j + offsetac ] += alpha*d[ (i+offsetbr)*col2 + j + offsetbc ];
	}
}
__kernel void axpy(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned char com, double alpha, double beta,
	unsigned long col, unsigned long col2,
	unsigned long offsetar, unsigned long offsetac,
	unsigned long offsetbr, unsigned long offsetbc){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	double aa, bb, cc, dd;
	if (com){
		aa = a[ (i+offsetar)*col + j + offsetac ]; bb = b[ (i+offsetar)*col + j + offsetac ];
		cc = c[ (i+offsetbr)*col2 + j + offsetbc ]; dd = d[ (i+offsetbr)*col2 + j + offsetbc ];

		c[ (i+offsetbr)*col2 + j + offsetbc ] = alpha*aa - beta*bb + cc;
		d[ (i+offsetbr)*col2 + j + offsetbc ] = alpha*bb + beta*aa + dd;
	}else{
		c[ (i+offsetbr)*col2 + j + offsetbc ] += alpha*a[ (i+offsetar)*col + j + offsetac ];
	}
}
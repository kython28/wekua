/*
	I know that this implementation of the matrix product isn't the most recommended,
	i know how to implement the optimized method with shared memory. Wekua wants to work
	in as many environments and processing devices as possible, many of them are old and
	with little storage. The optimized implementations of matrix products are made
	for predefined sizes of matrices, therefore it is not feasible when wekua is supposed to
	allow the operation of matrices of any size.

	The model as wekua is made is with kernels compiled at all working time, they aren't
	compiled when an operation is called. I have seen CLBLAS code, but looking i see
	that every time i want to perform an operation it compiles the kernel every time, that is
	not feasible. However, if you know how to implement a better method, i'm all ears: kike28.py@protonmail.ch
*/

__kernel void gemm(
	__global double *ar, __global double *ai,
	__global double *br, __global double *bi,
	__global double *cr, __global double *ci,
	unsigned char a_trans, unsigned char b_trans,
	double ralpha, double ialpha,
	double rbeta, double ibeta,
	unsigned long col, unsigned long rcol,
	unsigned long col2, unsigned long row,
	unsigned long offsetar, unsigned long offsetac,
	unsigned long offsetbr, unsigned long offsetbc,
	unsigned long offsetcr, unsigned long offsetcc,
	unsigned char com, unsigned long col3
){
	unsigned long i = get_global_id(0);
	unsigned long k = get_global_id(1);
	unsigned ccurr = ( i + offsetcr )*col3 + k + offsetcc;

	double ra = 0.0, rb = 0.0;
	double aa, bb, cc, dd;

	if (com){
		if (a_trans && b_trans){
			for (unsigned long j=0; j<row; j++){
				aa = ar[ (j+offsetar)*col + i + offsetac ]; bb = ai[ (j+offsetar)*col + i + offsetac ];
				cc = br[ (k+offsetbr)*col2 + j + offsetbc ]; dd = bi[ (k+offsetbr)*col2 + j + offsetbc ];

				ra += aa*cc - bb*dd;
				rb += aa*dd + bb*cc;
			}
		}else if (a_trans && b_trans == 0){
			for (unsigned long j=0; j<row; j++){
				aa = ar[ (j+offsetar)*col + i + offsetac ]; bb = ai[ (j+offsetar)*col + i + offsetac ];
				cc = br[ (j+offsetbr)*col2 + k + offsetbc ]; dd = bi[ (j+offsetbr)*col2 + k + offsetbc ];

				ra += aa*cc - bb*dd;
				rb += aa*dd + bb*cc;
			}
		}else if (a_trans == 0 && b_trans){
			for (unsigned long j=0; j<col; j++){
				aa = ar[ (i+offsetar)*col + j + offsetac ]; bb = ai[ (i+offsetar)*col + j + offsetac ];
				cc = br[ (k+offsetbr)*col2 + j + offsetbc ]; dd = bi[ (k+offsetbr)*col2 + j + offsetbc ];

				ra += aa*cc - bb*dd;
				rb += aa*dd + bb*cc;
			}
		}else if (a_trans == 0 && b_trans == 0){
			for (unsigned long j=0; j<col; j++){
				aa = ar[ (i+offsetar)*col + j + offsetac ]; bb = ai[ (i+offsetar)*col + j + offsetac ];
				cc = br[ (j+offsetbr)*col2 + k + offsetbc ]; dd = bi[ (j+offsetbr)*col2 + k + offsetbc ];

				ra += aa*cc - bb*dd;
				rb += aa*dd + bb*cc;
			}
		}
		aa = cr[ccurr]; bb = ci[ccurr];

		cr[ccurr] = ra*ralpha - rb*ialpha + aa*rbeta - bb*ibeta;
		ci[ccurr] = ra*ialpha + rb*ralpha + aa*ibeta + bb*rbeta;
	}else{
		if (a_trans && b_trans){
			for (unsigned long j=0; j<row; j++){
				ra += ar[ (j+offsetar)*rcol + i + offsetac ]*br[ (k+offsetbr)*col2 + j + offsetbc ];
			}
		}else if (a_trans && b_trans == 0){
			for (unsigned long j=0; j<row; j++){
				ra += ar[ (j+offsetar)*rcol + i + offsetac ]*br[ (j+offsetbr)*col2 + k + offsetbc ];
			}
		}else if (a_trans == 0 && b_trans){
			for (unsigned long j=0; j<col; j++){
				ra += ar[ (i+offsetar)*rcol + j + offsetac ]*br[ (k+offsetbr)*col2 + j + offsetbc ];
			}
		}else{
			for (unsigned long j=0; j<col; j++){
				ra += ar[ (i+offsetar)*rcol + j + offsetac ]*br[ (j+offsetbr)*col2 + k + offsetbc ];
			}
		}
		cr[ccurr] = ra*ralpha + rbeta*cr[ccurr];
	}
}
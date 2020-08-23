/*
	I know that this implementation of the matrix product isn't the most recommended,
	i know how to implement the optimized method with shared memory. Wekua wants to work
	in as many environments and processing devices as possible, many of them are old and
	with little storage, why this? Optimized implementations of matrix products are made
	for defined sizes of matrices, therefore it is not feasible when wekua is supposed to
	allow the operation of matrices of any size.

	The model as wekua is made is with kernels compiled at all working time, they aren't
	compiled when an operation is called. I have seen CLBLAS code, but looking i see
	that every time i want to perform an operation it compiles the kernel every time, that is
	not feasible. However, if you know how to implement a better method, i'm all ears: kike28.py@protonmail.ch
*/


__kernel void product(__global double *a, __global double *b,
	__global double *c, __global double *d,
	__global double *e, __global double *f,
	unsigned char com, unsigned long col, unsigned long col2){
	unsigned long i = get_global_id(0);
	unsigned long k = get_global_id(1);
	double ra = 0.0, rb, aa, bb, cc, dd;
	if (com){
		rb = 0.0;
		for (unsigned long j=0; j<col; j++){
			aa = a[i*col+j]; bb = b[i*col+j];
			cc = c[j*col2+k]; dd = d[j*col2+k];
			ra += aa*cc-bb*dd;
			rb += aa*dd+bb*cc;
		}
		f[i*col2+k] = rb;
	}else{
		for (unsigned long j=0; j<col; j++){
			ra += a[i*col+j]*c[j*col2+k];
		}
	}
	e[i*col2+k] = ra;
}
void gauss_real(__global double *a, __global double *b, unsigned long k, unsigned long i, unsigned long col){
	if (isnotequal(a[i*col+k], 0.0)){
		double aa=a[k*col+k]/a[i*col+k], bb;
		b[i*col+k] /= aa;
		for (unsigned long j=k; j<col; j++){
			bb = a[i*col+j];
			bb *= aa;
			bb -= a[k*col+j];
			a[i*col+j] = bb;
		}
	}
}

void gauss_com(__global double *a, __global double *b, __global double *c, __global double *d, unsigned long k, unsigned long i, unsigned long col){
	double aa, bb, cc, dd, akk, bkk, akj, bkj;
	aa = a[i*col+k]; bb = b[i*col+k];
	if (aa != 0.0 || bb != 0.0){
		for (unsigned long j=k; j<col; j++){
			cc=a[i*col+j]; dd=b[i*col+j];
			akk = a[k*col+k];
			bkk = b[k*col+k];
			akj = a[k*col+j];
			bkj = b[k*col+j];

			a[i*col+j] = (cc*akk-dd*bkk)-(akj*aa-bkj*bb);
			b[i*col+j] = (cc*bkk+dd*akk)-(akj*bb+bkj*aa);
		}
	}
}

__kernel void det(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long k, unsigned long col,
	unsigned char com){
	unsigned long i = get_global_id(0);

	c[k*col+k] = a[k*col+k];
	if (i > k){
		if (com){
			gauss_com(a, b, c, d, k, i, col);
			d[k*col+k] = b[k*col+k];
		}else{
			gauss_real(a, c, k, i, col);
		}
	}
}
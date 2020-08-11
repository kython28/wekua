void gauss_real(__global double *a, __global double *b, unsigned int k, unsigned int i, unsigned int col){
	if (isnotequal(a[i*col+k], 0.0)){
		double aa=a[k*col+k]/a[i*col+k];
		b[i*col+k] /= aa;
		for (unsigned int j=k; j<col; j++){
			a[i*col+j] = a[i*col+j]*aa - a[k*col+j];
		}
	}
}

void gauss_com(__global double *a, __global double *b, __global double *c, __global double *d, unsigned int k, unsigned int i, unsigned int col){
	if (a[i*col+k] != 0.0 || b[i*col+k] != 0.0){
		double aa = a[i*col+k], bb = b[i*col+k];
		for (unsigned int j=0; j<col; j++){
			double cc=a[i*col+j], dd=b[i*col+j];
			a[i*col+j] = (cc*a[k*col+k]-dd*b[k*col+k])-(a[k*col+j]*aa-b[k*col+j]*bb);
			b[i*col+j] = (cc*b[k*col+k]+dd*a[k*col+k])-(a[k*col+j]*bb+b[k*col+j]*aa);
		}
	}
}

__kernel void det(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int k, unsigned int col,
	unsigned char com){
	unsigned int i = get_global_id(0);

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
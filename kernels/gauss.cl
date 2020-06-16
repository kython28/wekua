void inv_real(__global double *a, __global double *b, unsigned int k, unsigned int i, unsigned int col, unsigned char otherm){
	if (a[i*col+k] != 0.0){
		double aa=a[i*col+k];
		for (unsigned int j=0; j<col; j++){
			a[i*col+j] = a[i*col+j]*a[k*col+k] - a[k*col+j]*aa;
			if (otherm){
				b[i*col+j] = b[i*col+j]*a[k*col+k] - b[k*col+j]*aa;
			}
		}
	}
}

void inv_com(__global double *a, __global double *b, __global double *c, __global double *d, unsigned int k, unsigned int i, unsigned int col, unsigned char otherm){
	if (a[i*col+k] != 0.0 || b[i*col+k] != 0.0){
		double aa = a[i*col+k], bb = b[i*col+k];
		for (unsigned int j=0; j<col; j++){
			double cc=a[i*col+j], dd=b[i*col+j];
			a[i*col+j] = (cc*a[k*col+k]-dd*b[k*col+k])-(a[k*col+j]*aa-b[k*col+j]*bb);
			b[i*col+j] = (cc*b[k*col+k]+dd*a[k*col+k])-(a[k*col+j]*bb+b[k*col+j]*aa);
			if (otherm){
				cc=c[i*col+j]; dd=d[i*col+j];
				c[i*col+j] = (cc*c[k*col+k]-dd*d[k*col+k])-(c[k*col+j]*aa-d[k*col+j]*bb);
				d[i*col+j] = (cc*d[k*col+k]+dd*c[k*col+k])-(c[k*col+j]*bb+d[k*col+j]*aa);
			}
		}
	}
}

__kernel void gauss(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned char com,
	unsigned char otherm, unsigned char t){
	unsigned int k = get_global_id(0);
	unsigned int i = get_global_id(1);

	if (t){
		i = col-i-1;
		k = col-k-1;
	}

	if ((i > k && t == 0) || (i < k && t)){
		if (com){
			inv_com(a, b, c, d, k, i, col, otherm);
		}else{
			inv_real(a, c, k, i, col, otherm);
		}
	}
}
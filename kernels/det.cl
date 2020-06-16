void det_real(__global double *a, __global double *b, unsigned int k, unsigned int i, unsigned int col){
	if (a[i*col+k] != 0.0){
		double aa=a[i*col+k];
		for (unsigned int j=k; j<col; j++){
			a[i*col+j] = a[i*col+j]*a[k*col+k] - a[k*col+j]*aa;
		}
		if (a[k*col+k] != 0.0){
			b[i*col+k] = 1/a[k*col+k];
		}else{
			b[i*col+k] = 0.0;
		}
	}
}

void det_com(__global double *a, __global double *b, __global double *c, __global double *d, unsigned int k, unsigned int i, unsigned int col){
	double r, ang;
	if (a[i*col+k] != 0.0 || b[i*col+k] != 0.0){
		double aa = a[i*col+k], bb = b[i*col+k];
		for (unsigned int j=k; j<col; j++){
			double cc=a[i*col+j], dd=b[i*col+j];
			a[i*col+j] = (cc*a[k*col+k]-dd*b[k*col+k])-(a[k*col+j]*aa-b[k*col+j]*bb);
			b[i*col+j] = (cc*b[k*col+k]+dd*a[k*col+k])-(a[k*col+j]*bb+b[k*col+j]*aa);
			
		}
		if (a[k*col+k] != 0.0 || b[k*col+k] != 0.0){
			r = 1/sqrt(a[k*col+k]*a[k*col+k]+b[k*col+k]*b[k*col+k]);
			if (a[k*col+k] == 0.0){
				ang = M_PI_2;
			}else{
				ang = tanh(b[k*col+k]/a[k*col+k]);
			}
			c[i*col+k] = r*cos(-1.0*ang);
			d[i*col+k] = r*sin(-1.0*ang);
		}else{
			c[i*col+k] = 0.0;
			d[i*col+k] = 0.0;
		}
	}
}

__kernel void det(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned char com){
	unsigned int i = get_global_id(0);
	unsigned int k = get_global_id(1);

	if (i > k){
		if (com){
			det_com(a, b, c, d, k, i, col);
		}else{
			det_real(a, c, k, i, col);
		}
	}
	c[k*col+k] = a[k*col+k];
	if (com){
		d[k*col+k] = b[k*col+k];
	}
}
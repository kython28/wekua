__kernel void gauss2(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned char com){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);

	double ang, r, cc, dd;
	if (com){
		r = 1/sqrt(a[i*col+i]*a[i*col+i]+b[i*col+i]*b[i*col+i]);
		if (a[i*col+i] == 0.0){
			ang = M_PI_2;
		}else{
			ang = tanh(b[i*col+i]/a[i*col+i]);
		}
		cc = c[i*col+j]; dd = d[i*col+j];
		c[i*col+j] = r*cos(-1.0*ang)*cc-r*sin(-1.0*ang)*dd;
		d[i*col+j] = r*cos(-1.0*ang)*dd+r*sin(-1.0*ang)*cc;
	}else{
		c[i*col+j] *= 1/a[i*col+i];
	}
}
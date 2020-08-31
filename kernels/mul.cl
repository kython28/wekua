__kernel void mul(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long col, unsigned char com,
	unsigned long offsetar, unsigned long offsetac){
	unsigned long i = get_global_id(0);

	double aa, bb, cc, dd;
	unsigned long ee;
	if (com){
		for (unsigned long j=0; j<col; j++){
			ee = (i+offsetar)*col+j+offsetac;
			aa = a[ee]; bb = b[ee];
			cc = c[i]; dd = d[i];
			c[i] = aa*cc - bb*dd;
			d[i] = aa*dd + bb*cc;
		}
	}else{
		for (unsigned long j=0; j<col; j++){
			c[i] *= a[i*col+j];
		}
	}
}
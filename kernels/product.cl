__kernel void product(__global double *a, __global double *b,
	__global double *c, __global double *d,
	__global double *e, __global double *f,
	unsigned char com, unsigned long col, unsigned long col2){
	unsigned long i = get_global_id(0);
	unsigned long k = get_global_id(1);
	double ra = 0.0, rb;
	if (com){
		rb = 0.0;
		for (unsigned long j=0; j<col; j++){
			ra += a[i*col+j]*c[j*col2+k]-b[i*col+j]*d[j*col2+k];
			rb += a[i*col+j]*d[j*col2+k]+b[i*col+j]*c[j*col2+k];
		}
		f[i*col2+k] = rb;
	}else{
		for (unsigned long j=0; j<col; j++){
			ra += a[i*col+j]*c[j*col2+k];
		}
	}
	e[i*col2+k] = ra;
}
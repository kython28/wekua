__kernel void product(__global double *a, __global double *b,
	__global double *c, __global double *d,
	__global double *e, __global double *f,
	unsigned char com, unsigned int col, unsigned int col2){
	unsigned int i = get_global_id(0);
	unsigned int k = get_global_id(1);
	if (com){
		for (unsigned int j=0; j<col; j++){
			e[i*col2+k] += a[i*col+j]*c[j*col2+k]-b[i*col+j]*d[j*col2+k];
			f[i*col2+k] += a[i*col+j]*d[j*col2+k]+b[i*col+j]*c[j*col2+k];
		}
	}else{
		for (unsigned int j=0; j<col; j++){
			e[i*col2+k] += a[i*col+j]*c[j*col2+k];	
		}
	}
}
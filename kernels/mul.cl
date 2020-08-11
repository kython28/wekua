__kernel void mul(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned char com){
	unsigned int i = get_global_id(0);

	if (com){
		for (unsigned int j=0; j<col; j++){
			c[i] = a[i*col+j]*c[i] - b[i*col+j]*d[i];
			d[i] = a[i*col+j]*d[i] + b[i*col+j]*c[i];
		}
	}else{
		for (unsigned int j=0; j<col; j++){
			c[i] *= a[i*col+j];
		}
	}
}
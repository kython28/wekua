__kernel void cut(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned int col, unsigned int x, unsigned int y,
	unsigned int w, unsigned char com){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	a[i*col+j] = c[(y+i)*w+x+j];
	if (com){
		b[i*col+j] = d[(y+i)*w+x+j];	
	}
}
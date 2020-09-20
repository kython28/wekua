__kernel void diag(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned char typ, unsigned char com){
	unsigned long i = get_global_id(0);
	unsigned long col = get_global_size(0);
	if (typ){
		a[i*col+i] = c[i];
		if (com){
			b[i*col+i] = d[i];
		}
	}else{
		a[i] = c[i*col+i];
		if (com){
			b[i] = d[i*col+i];
		}
	}
}
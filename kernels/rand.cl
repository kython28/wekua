__kernel void rand(__global double *a, __global double *b,
	__global long *c, __global long *d, unsigned char com){
	unsigned long i = get_global_id(0);
	double RECIP_BPF = 0.00000000000000011102230246251565404236316680908203125;
	a[i] = (c[i] >> 4)*RECIP_BPF;
	if (com){
		b[i] = (d[i] >> 4)*RECIP_BPF;
	}
}
__kernel void conv1d(__global double *a, __global double *b,
	__global double *c, __global double *d,
	__global double *k, __global double *ik,
	unsigned long klen, unsigned long col, unsigned long col2,
	unsigned char com){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);
	double ra = k[klen], rb, aa, bb, cc, dd;
	if (com){
		rb = ik[klen];
		for (unsigned long h=0; h<klen; h++){
			aa = c[i*col2+j+h]; bb = d[i*col2+j+h];
			cc = k[h]; dd = ik[h];
			ra += aa*cc - bb*cc;
			rb += aa*dd + bb*cc;
		}
		b[i*col+j] = rb;
	}else{
		for (unsigned long h=0; h<klen; h++){
			ra += c[i*col2+j+h]*k[h];
		}
	}
	a[i*col+j] = ra;
}
__kernel void uniform(__global double *a, __global double *b,
	double ra, double ia, double re, double ie,
	unsigned int col, unsigned char com){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);

	double aa, bb, cc, dd;
	if (com){
		aa = a[i*col+j]; bb = b[i*col+j];
		cc = re-ra;
		dd = ie-ia;
		a[i*col+j] = ra + aa*cc - bb*dd;
		b[i*col+j] = re + aa*dd + bb*cc;
	}else{
		a[i*col+j] = ra + (re-ra)*a[i*col+j];
	}
}
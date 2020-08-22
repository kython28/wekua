void gauss_real(__global double *a, __global double *b, unsigned long k, unsigned long i, unsigned long col, unsigned char otherm, unsigned char pn){
	if (isnotequal(a[i*col+i], 0.0) && !isnan(a[i*col+i])){
		if (isnotequal(a[i*col+k], 0.0)){
			double aa=a[k*col+k]/a[i*col+k], bb;
			for (unsigned long j=0; j<col; j++){
				bb = a[i*col+j];
				bb *= aa;
				bb -= a[k*col+j];
				a[i*col+j] = bb;
				if (otherm){
					bb = b[i*col+j];
					bb *= aa;
					bb -= b[k*col+j];
					b[i*col+j] = bb;
				}
			}
		}
	}else if (pn && !isnan(a[i*col+i])){
		for (unsigned long j=0; j<col; j++){
			a[i*col+j] = NAN;
			if (otherm){
				b[i*col+j] = NAN;
			}
		}
	}
}

void calc_coeff(double a, double b, double *c, double *d){
	double r, ang;

	r = a/(a*a+b*b);
	ang = -1.0*b/(a*a+b*b);

	a = c[0]*r - d[0]*ang;
	b = c[0]*ang + d[0]*r;

	c[0] = a;
	d[0] = b;
}

void gauss_com(__global double *a, __global double *b, __global double *c, __global double *d, unsigned long k, unsigned long i, unsigned long col, unsigned char otherm, unsigned char pn){
	double aa, bb, cc, dd;
	aa = a[i*col+i]; bb = b[i*col+i];
	if ((isnotequal(aa, 0.0) && !isnan(aa)) || (isnotequal(bb, 0.0) && !isnan(bb))){
		if (isnotequal(a[i*col+k], 0.0) || isnotequal(b[i*col+k], 0.0)){
			aa=a[k*col+k]; bb=b[k*col+k];
			calc_coeff(a[i*col+k], b[i*col+k], &aa, &bb);
			for (unsigned long j=0; j<col; j++){
				cc=a[i*col+j]; dd=b[i*col+j];
				a[i*col+j] = (cc*aa - dd*bb) - a[k*col+j];
				b[i*col+j] = (cc*bb + dd*aa) - b[k*col+j];
				if (otherm){
					cc=c[i*col+j]; dd=d[i*col+j];
					c[i*col+j] = (cc*aa - dd*bb) - c[k*col+j];
					d[i*col+j] = (cc*bb + dd*aa) - d[k*col+j];
				}
			}
		}
	}else if (pn && (!isnan(a[i*col+i]) || !isnan(b[i*col+i]))){
		for (unsigned long j=0; j<col; j++){
			a[i*col+j] = NAN;
			b[i*col+j] = NAN;
			if (otherm){
				c[i*col+j] = NAN;
				d[i*col+j] = NAN;
			}
		}
	}
}

__kernel void gauss(__global double *a, __global double *b,
	__global double *c, __global double *d,
	unsigned long k, unsigned long col,
	unsigned char com, unsigned char otherm,
	unsigned char t, unsigned char pn){
	unsigned long i = get_global_id(0);

	if (t){
		i = col-i-1;
		k = col-k-1;
	}

	if ((i > k && t == 0) || (i < k && t)){
		if (com){
			gauss_com(a, b, c, d, k, i, col, otherm, pn);
		}else{
			gauss_real(a, c, k, i, col, otherm, pn);
		}
	}
}
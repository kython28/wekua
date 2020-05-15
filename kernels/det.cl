__kernel void det(__global double *a, __global double *b, unsigned int c){
	unsigned int k = get_global_id(0);
	unsigned int i = get_global_id(1);
	double aa=1.0, bb=1.0;
	if (i > k && k+1 < c){
		if (a[k*c+k]*a[i*c+k] > 0){
			if (a[k*c+k] < 0){
				aa = -1.0;
			}else{
				bb = -1.0;
			}
		}else{
			bb = -1.0;
		}
		bb *= a[i*c+k];
		for (unsigned int j=0; j<c; j++){
			a[i*c+j] = aa*a[k*c+k]*a[i*c+j] + a[k*c+j]*bb;
		}
		b[i*c+k] = 1/a[k*c+k];
	}
	b[k*c+k] = a[k*c+k];
}

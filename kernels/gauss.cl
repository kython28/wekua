__kernel void gauss(__global double *a, __global double *b, unsigned int t, unsigned int c){
	unsigned int k = get_global_id(0);
	unsigned int i = get_global_id(1);
	double aa=1.0, bb=1.0;
	if (i > k && k+1 < c){
		if (t != 0){
			k = c-k-1;
			i = c-i-1;
		}
		if (a[k*c+k]*a[i*c+k] > 0){
			if (a[k*c+k] < 0){
				aa = -1.0;
			}else{
				bb = -1.0;
			}
		}else{
			bb = -1.0;
		}
		aa *= a[k*c+k];
		bb *= a[i*c+k];
		for (unsigned int j=0; j<c; j++){
			b[i*c+j] = aa*b[i*c+j] + b[k*c+j]*bb;
			a[i*c+j] = aa*a[i*c+j] + a[k*c+j]*bb;	
		}
	}
}

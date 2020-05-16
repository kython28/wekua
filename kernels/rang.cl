__kernel void rang(__global double *a, unsigned int c, unsigned int r){
	unsigned int k = get_global_id(0);
	unsigned int i = get_global_id(1);
	double aa=1.0, bb=1.0;
	if (i > k && k < c){
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
		for (unsigned int j=0; j<r; j++){
			a[i*c+j] = aa*a[i*c+j] + a[k*c+j]*bb;	
		}
	}
}

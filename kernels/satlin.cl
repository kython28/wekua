__kernel void satlin(__global double *a, __global double *b,
	unsigned char com){
	unsigned long i = get_global_id(0);

	if (a[i] < 0.0){
		a[i] = 0.0;
		if (com){
			b[i] = 0.0;
		}
	}else if (com){
		if (sqrt(a[i]*a[i] + b[i]*b[i]) > 1.0){
			a[i] = 1.0;
			b[i] = 0.0;
		}
	}else{
		if (a[i] > 1.0){
			a[i] = 1.0;
		}
	}
}
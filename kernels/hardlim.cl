__kernel void hardlim(__global double *a, __global double *b,
	unsigned char com){
	unsigned long i = get_global_id(0);
	
	if (a[i] >= 0.0){
		a[i] = 1.0;
	}else{
		a[i] = 0.0;
	}

	if (com){
		b[i] = 0.0;
	}
}
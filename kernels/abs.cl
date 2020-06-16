__kernel void absolute(__global double *a, __global double *b,
	unsigned int c, unsigned char com){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	if (com){
		a[i*c+j] = sqrt(a[i*c+j]*a[i*c+j]+b[i*c+j]*b[i*c+j]);
	}else{
		if (a[i*c+j] < 0.0){
			a[i*c+j] *= -1.0;
		}
	}
}
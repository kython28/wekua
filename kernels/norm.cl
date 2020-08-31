__kernel void norm(__global double *a, __global double *b,
	unsigned char com, unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	double aa;
	if (com){
		aa = a[i];
		a[i] = aa*aa-b[i]*b[i];
		b[i] *= 2*aa;
	}else{
		a[i] *= a[i];
	}
}
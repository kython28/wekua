__kernel void absolute(__global double *a, __global double *b,
	unsigned char com){
	unsigned long i = get_global_id(0);
	double aa, bb;
	if (com){
		aa = a[i]; bb = b[i];
		a[i] = sqrt(aa*aa+bb*bb);
	}else{
		a[i] = fabs(a[i]);
	}
}
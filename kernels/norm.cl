__kernel void norm(__global double *a, __global double *b,
	unsigned char com, unsigned long col){
	unsigned long i = get_global_id(0)*col+get_global_id(1);
	double aa = a[i], bb;
	if (com){
		bb = b[i];
		a[i] = aa*aa-bb*bb;
		b[i] = bb*2*aa;
	}else{
		a[i] = aa*aa;
	}
}
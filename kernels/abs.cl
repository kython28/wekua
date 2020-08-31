__kernel void absolute(__global double *a, __global double *b,
	unsigned char com, unsigned long col){
	unsigned long current = get_global_id(0)*col+get_global_id(1);
	double aa, bb;
	if (com){
		aa = a[current]; bb = b[current];
		a[current] = sqrt(aa*aa+bb*bb);
	}else{
		a[current] = fabs(a[current]);
	}
}
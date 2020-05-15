__kernel void absolute(__global double *a){
	unsigned int i = get_global_id(0);
	if (a[i] < 0){
		a[i] = -1.0*a[i];
	}
}

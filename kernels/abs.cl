__kernel void absolute(__global double *a, unsigned int c){
	unsigned int y = get_global_id(0);
	unsigned int x = get_global_id(1);
	if (a[y*c+x] < 0){
		a[y*c+x] *= -1.0;
	}
}

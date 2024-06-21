__kernel void arange(__global double *a, __global double *b,
	double start_r, double start_i,
	double delta_r, double delta_i,
	double delta, double ang,
	unsigned long col, unsigned char trans
){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);
	unsigned long posi = i*col+j, x, y;

	if (trans){
		x = convert_ulong_sat_rte(fabs(i*delta_r*cos(ang)/delta));
		y = convert_ulong_sat_rte(fabs(i*delta_i*sin(ang)/delta));
	}else{
		x = convert_ulong_sat_rte(fabs(j*delta_r*cos(ang)/delta));
		y = convert_ulong_sat_rte(fabs(j*delta_i*sin(ang)/delta));
	}

	a[posi] = start_r + x*delta_r;
#if com
	b[posi] = start_i + y*delta_i;
#endif
}
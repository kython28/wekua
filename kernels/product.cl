__kernel void product(__global double *a, __global double *b, __global double *c, unsigned int co, unsigned int ko){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	unsigned int k = get_global_id(2);
	c[i*ko+k] += a[i*co+j]*b[j*ko+k];
}

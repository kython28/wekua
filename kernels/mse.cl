#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void mse(__global wk *tr, __global wk *ti,
	__global wk *ar, __global wk *ai,
	__global wk *er, __global wk *ei,
	__global wk *devr, __global wk *devi,
	unsigned long col, unsigned char dev, unsigned char com){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

	wk error, errori;
	error = tr[i] - ar[i];
	if (com){
		errori = ti[i] - ai[i];
		if (dev){
			devr[i] = -2*error;
			devi[i] =  -2*errori;
		}
		complex_mul(&error, &errori, error, errori);
		er[i] = error;
		ei[i] = errori;
	}else{
		er[i] = error*error;
		if (dev){
			devr[i] = -2*error;
		}
	}
}
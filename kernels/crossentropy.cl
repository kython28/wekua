#include "/usr/lib/wekua_kernels/dtype.cl"

#define LN_2 0.69314718055994528622676398299518041312694549560546875

__kernel void crossentropy(__global wk *tr, __global wk *ti,
	__global wk *ar, __global wk *ai,
	__global wk *er, __global wk *ei,
	__global wk *devr, __global wk *devi,
	unsigned long col, unsigned char dev){
	unsigned long i = get_global_id(0)*col+get_global_id(1);

	wk error = ar[i];
	wk wt = tr[i];
#if com
	wk errori = ai[i];
	wk wti = ti[i];
	if (dev){
		devr[i] = -2*error;
		devi[i] =  -2*errori;
	}
	complex_mul(&error, &errori, error, errori);
	er[i] = error;
	ei[i] = errori;
#else
	if (wt == 0){
		error = 1.0 - error;
	}
	er[i] = -log2(error);
	if (dev){
		devr[i] = -1.0/(error*LN_2);
	}
#endif
}
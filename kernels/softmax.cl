#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void softmax(__global wks *a, unsigned long col, unsigned long stride){

	unsigned long i = get_global_id(0);
#if com
	
#else
	wks total = 0;
	for (unsigned long x=0; x<col; x++){
		total += exp(a[i*stride + x]);
	}

	for (unsigned long x=0; x<col; x++){
		a[i*stride + x] = exp(a[i*stride + x]) / total;
	}
#endif
}

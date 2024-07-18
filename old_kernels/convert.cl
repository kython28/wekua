#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void convert(__global wks *a, __global wks *b,
	__global void *c, __global void *d,
	unsigned long col, unsigned long col2,
	unsigned char dt){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);

	if (dt == 0){
		a[i*col+j] = convert_T(((__global char*)c)[i*col2+j]);
	}else if (dt == 1){
		a[i*col+j] = convert_T(((__global unsigned char*)c)[i*col2+j]);
	}else if (dt == 2){
		a[i*col+j] = convert_T(((__global short*)c)[i*col2+j]);
	}else if (dt == 3){
		a[i*col+j] = convert_T(((__global unsigned short*)c)[i*col2+j]);
	}else if (dt == 4){
		a[i*col+j] = convert_T(((__global int*)c)[i*col2+j]);
	}else if (dt == 5){
		a[i*col+j] = convert_T(((__global unsigned int*)c)[i*col2+j]);
	}else if (dt == 6){
		a[i*col+j] = convert_T(((__global long*)c)[i*col2+j]);
	}else if (dt == 7){
		a[i*col+j] = convert_T(((__global unsigned long*)c)[i*col2+j]);
	}else if (dt == 8){
		a[i*col+j] = convert_T(((__global float*)c)[i*col2+j]);
	}else if (dt == 9){
		a[i*col+j] = convert_T(((__global double*)c)[i*col2+j]);
	}

#if com
	if (dt == 0){
		b[i*col+j] = convert_T(((__global char*)d)[i*col2+j]);
	}else if (dt == 1){
		b[i*col+j] = convert_T(((__global unsigned char*)d)[i*col2+j]);
	}else if (dt == 2){
		b[i*col+j] = convert_T(((__global short*)d)[i*col2+j]);
	}else if (dt == 3){
		b[i*col+j] = convert_T(((__global unsigned short*)d)[i*col2+j]);
	}else if (dt == 4){
		b[i*col+j] = convert_T(((__global int*)d)[i*col2+j]);
	}else if (dt == 5){
		b[i*col+j] = convert_T(((__global unsigned int*)d)[i*col2+j]);
	}else if (dt == 6){
		b[i*col+j] = convert_T(((__global long*)d)[i*col2+j]);
	}else if (dt == 7){
		b[i*col+j] = convert_T(((__global unsigned long*)d)[i*col2+j]);
	}else if (dt == 8){
		b[i*col+j] = convert_T(((__global float*)d)[i*col2+j]);
	}else if (dt == 9){
		b[i*col+j] = convert_T(((__global double*)d)[i*col2+j]);
	}
#endif
}
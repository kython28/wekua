#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void linear_bias_step(
	__global wk *ar, __global wk *ai,
	__global wks *br, __global wks *bi,

	unsigned long col, unsigned char com
){
	unsigned long j = (get_global_id(0) << 1);
	unsigned long arow = j*col;

	wk C11, C12;

	#if width == 1
	C11 = 0; C12 = 0;
	#else
	C11 = (wk)(0); C12 = (wk)(0);
	#endif

	for (unsigned long k = 0; k < col2; k += 2){
		C11 += ar[arow + k] + ar[arow + k + 1];
		C12 += ar[(arow << 1) + k] + ar[(arow << 1) + k + 1];
	}

	#if width == 1
	br[j] = C11; br[j + 1] = C12;
	#else
	br[j] = sum(C11); br[j + 1] = sum(C12);
	#endif

	if (com){
		#if width == 1
		C11 = 0; C12 = 0;
		#else
		C11 = (wk)(0); C12 = (wk)(0);
		#endif

		for (unsigned long k = 0; k < col2; k += 2){
			C11 += ai[arow + k] + ai[arow + k + 1];
			C12 += ai[(arow << 1) + k] + ai[(arow << 1) + k + 1];
		}

		#if width == 1
		bi[j] = C11; bi[j + 1] = C12;
		#else
		bi[j] = sum(C11); bi[j + 1] = sum(C12);
		#endif
	}
}
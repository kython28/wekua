#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void l1regu(
	__global wk *ar, __global wk *ai,
	ulong col
){
	ulong i = get_global_id(0)*col + get_global_id(1);

#if com
	wk a = ar[i];
	wk b = ai[i];
	wk c = sqrt(a*a + b*b);

	ar[i] = a/(c + FLT_EPSILON);
	ai[i] = b/(c + FLT_EPSILON);
#else
	ar[i] = 1.0*sign(ar[i]);
#endif
}
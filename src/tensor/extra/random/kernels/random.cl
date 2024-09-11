#include "wekua.h"

#if dtype == 8
#define random_wks uint
#define RAND_BASE 4294967295.0
#else
#define random_wks ulong
#define RAND_BASE 18446744073709551615.0
#endif

__kernel void random(
	__global random_wks *random_numbers, __global wks *numbers,
	const ulong row_pitch, const ulong col
) {
	const ulong i = get_global_id(0);
	const ulong j = get_global_id(1);

#if dtype >= 8
#if com == 1
	const ulong index = i*row_pitch + j*2;

	if (j >= col) {
		numbers[index] = 0.0;
		numbers[index + 1] = 0.0;
	}else{
		numbers[index] = random_numbers[index]/RAND_BASE;
		numbers[index + 1] = random_numbers[index + 1]/RAND_BASE;
	}
#else
	const ulong index = i*row_pitch + j;
	if (j >= col) {
		numbers[index] = 0.0;
	}else{
		numbers[index] = random_numbers[index]/RAND_BASE;
	}
#endif
#else
#if com == 1
	const ulong index = i*row_pitch + j*2;

	if (j >= col) {
		numbers[index] = 0;
		numbers[index + 1] = 0;
	}
#else
	const ulong index = i*row_pitch + j;
	if (j >= col) {
		numbers[index] = 0;
	}
#endif
#endif
}

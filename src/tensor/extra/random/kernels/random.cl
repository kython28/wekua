#include "wekua.h"

#if dtype == 8
#define random_wks uint
#else
#define random_wks ulong
#endif

__kernel void random(
	__global random_wks *random_numbers,
	__global wks *numbers
) {
	ulong tid = get_global_id(0);
	numbers[tid] = random_numbers[tid]/100000.0;
}

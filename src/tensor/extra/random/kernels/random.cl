#include "wekua.h"

#if dtype == 8
#define random_wks uint
#define RAND_BASE 4294967295.0
#else
#define random_wks ulong
#define RAND_BASE 18446744073709551615.0
#endif

__kernel void random(
	__global random_wks *random_numbers,
	__global wks *numbers
) {
	ulong i = get_global_id(0);
	numbers[i] = random_numbers[i]/RAND_BASE;
}

#include "../headers/tensor.h"
#include <math.h>

void get_local_work_items(uint64_t *global_work_items, uint64_t *local_work_items, uint64_t ndim, uint64_t max){
	uint64_t maximum_items_per_dimension = (uint64_t)(pow(1.0*max, 1.0/ndim));
	for (uint32_t j=0; j<ndim; j++){
		register uint64_t g_work_items = global_work_items[j];
		if (g_work_items < maximum_items_per_dimension){
			local_work_items[j] = g_work_items;
			continue;
		}
		register uint64_t l_work_items = maximum_items_per_dimension;
		while (g_work_items%l_work_items) l_work_items--;
		local_work_items[j] = l_work_items;
	}
}

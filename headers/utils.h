#ifndef WEKUA_UTILS_H
#define WEKUA_UTILS_H

#include <stdint.h>

void get_local_work_items(uint64_t *global_work_items, uint64_t *local_work_items, uint64_t ndim, uint64_t max);

#endif
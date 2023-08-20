#ifndef WEKUA_UTILS_H
#define WEKUA_UTILS_H

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdint.h>

void get_local_work_items(uint64_t *global_work_items, uint64_t *local_work_items, uint64_t ndim, uint64_t max);
void wait_for_and_release_cl_events(cl_event *events, const uint32_t n_events);

#endif
#ifndef CACHE_H
#define CACHE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _w_cache {
	uint64_t ndata;
	void *data;
} *wcache;

#ifdef __cplusplus
}
#endif
#endif
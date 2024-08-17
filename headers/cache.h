#ifndef WEKUA_CACHE_H
#define WEKUA_CACHE_H

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


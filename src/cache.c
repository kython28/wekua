#include "wekua.h"

void wekuaCacheFree(wcache cache, uint32_t nw, cl_event *be){
	uint64_t ndata = cache->ndata;
	wmatrix *data = cache->data;

	clWaitForEvents(nw, be);

	for (uint64_t x=0; x<ndata; x++){
		wekuaFreeMatrix(data[x], 0, NULL);
	}

	free(cache);
}
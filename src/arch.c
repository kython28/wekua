#include "wekua.h"

warch *wekuaArch(wekuaContext *ctx, uint32_t nmodule, wmatrix *(*func)(wmodule **, uint32_t, wmatrix *), uint8_t com){
	if (ctx == NULL){
		return NULL;
	}
	warch *a = (warch*) calloc(1, sizeof(warch));
	a->nmodule[0] = nmodule;
	a->nmodule[1] = 0;
	a->nmodule[2] = 0;
	a->modules = (wmodule**) calloc(nmodule, sizeof(wmodule*));
	a->com = com;
	a->func = func;
	a->cache = NULL;
	a->pseq = 0;
	return a;
}

void addModuleToArch(warch *arch, wmodule *module){
	if (arch == NULL || module == NULL){
		return;
	}else if (arch->nmodule[1]+1 > arch->nmodule[0]){
		return;
	}
	arch->modules[arch->nmodule[1]] = module;
	arch->nmodule[1]++;
	arch->nmodule[2] += module->nmod;	
}

void configureWekuaArch(warch *arch){
	if (arch->cache != NULL){
		for (uint32_t x=0; x<=arch->pseq; x++){
			wekuaFreeMatrix(arch->cache[x], 0, NULL);
		}
		free(arch->cache);
	}
	arch->cache = calloc(arch->nmodule[2]+1, sizeof(wmatrix*));
	arch->weight = calloc(arch->nmodule[2], sizeof(wmatrix*));
	arch->w_id = calloc(arch->nmodule[2], 8);
	arch->s = calloc(arch->nmodule[2], sizeof(wmatrix*));
	arch->acti_funcs = calloc(arch->nmodule[2], sizeof(wacti*));
	arch->pseq = 0;
	uint64_t j = 0;
	for (uint32_t x=0; x<arch->nmodule[1]; x++){
		arch->modules[x]->set_cache_id(arch->modules[x], j, arch->cache, &arch->weight[j], &arch->pseq, arch->w_id, arch->acti_funcs);
		j += (int64_t)arch->modules[x]->nmod;
	}
}

wmatrix *runWekuaArch(warch *arch, wmatrix *input, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);

	for (uint32_t x=0; x<arch->pseq; x++){
		wekuaFreeMatrix(arch->cache[x], 0, NULL);
		arch->cache[x] = NULL;
	}
	arch->pseq = 0;
	return arch->func(arch->modules, arch->nmodule[1], input);
}

void wekuaFreeArch(warch *arch, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	for (uint32_t x=0; x<arch->nmodule[0]; x++){
		arch->modules[x]->free_func(arch->modules[x], 0, NULL);
	}
	if (arch->pseq > 0){
		for (uint32_t x=0; x<arch->pseq-1; x++){
			wekuaFreeMatrix(arch->s[x], 0, NULL);
			wekuaFreeMatrix(arch->cache[x], 0, NULL);
		}
		wekuaFreeMatrix(arch->cache[arch->pseq-1], 0, NULL);
	}
	free(arch->cache);
	free(arch->weight);
	free(arch->modules);
	free(arch->w_id);
	free(arch->s);
	free(arch->acti_funcs);
}
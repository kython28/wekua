#include "wekua.h"

void seq_set_cache_id(void *m, int64_t id, void *cache, void *w, uint32_t *pseq, uint8_t *fa){
	wmodule *module = m;
	module->arch_id = id;
	for (uint32_t y=1; y <= ((uint32_t*)module->data[0])[1]; y++){
		((wmodule*)module->data[y])->set_cache_id(((wmodule*)module->data[y]), id, cache, w, pseq, fa);
		id += (int64_t)((wmodule*)module->data[y])->nmod;
	}
}

wmodule *wekuaSequential(wekuaContext *ctx, uint32_t nmodule, uint8_t com){
	if (ctx == NULL){
		return NULL;
	}
	wmodule *seq = (wmodule*) calloc(1, sizeof(wmodule));
	seq->data = (void **) calloc(nmodule+1, sizeof(void*));
	seq->data[0] = (void*) calloc(2, 4);
	((uint32_t*)seq->data[0])[0] = nmodule;
	((uint32_t*)seq->data[0])[1] = 0;
	seq->com = com;
	seq->func = &runWekuaSequential;
	seq->free_func = &freeWekuaSequential;
	seq->set_cache_id = &seq_set_cache_id;
	seq->nmod = 0;
	return seq;
}

void addModuleToSequential(wmodule *sequential, wmodule *module){
	if (sequential == NULL || module == NULL){
		return;
	}else if (((uint32_t*)sequential->data[0])[1]+1 > ((uint32_t*)sequential->data[0])[0]){
		return;
	}
	sequential->data[((uint32_t*)sequential->data[0])[1]+1] = module;
	((uint32_t*)sequential->data[0])[1]++;
	sequential->nmod += module->nmod;
}

wmatrix *runWekuaSequential(void *m, wmatrix *input){
	if (m == NULL || input == NULL){
		return NULL;
	}
	wmodule *module = m;
	wmatrix *output, **tmp;
	tmp = (wmatrix**) calloc(2, sizeof(wmatrix*));
	uint8_t d = 1;
	tmp[0] = ((wmodule*)module->data[1])->func(((wmodule*)module->data[1]), input);
	for (uint32_t x=2; x <= ((uint32_t*)module->data[0])[1]; x++){
		tmp[d] = ((wmodule*)module->data[x])->func(((wmodule*)module->data[x]), tmp[d^1]);
		d ^= 1;
		if (module->arch_id < 0){
			wekuaFreeMatrix(tmp[d]);
		}
	}
	output = tmp[d^1];
	free(tmp);
	return output;
}

void freeWekuaSequential(void *m){
	wmodule *seq = m;
	for (uint32_t x=1; x <= ((uint32_t*)seq->data[0])[1]; x++){
		((wmodule*)seq->data[x])->free_func(seq->data[x]);
	}
	free(seq->data[0]);
	free(seq->data);
	free(m);
}
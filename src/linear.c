#include "wekua.h"

void linear_set_cache_id(void *m, int64_t id, void *cache, void *w, uint32_t *pseq, int64_t *w_id, wacti **acti){
	wmodule *module = m;
	wmatrix **weig = w;
	
	module->arch_id = id;
	module->cache = cache;
	module->pseq = pseq;
	module->w_id = w_id;
	for (uint32_t x=0; x<module->nmod; x++){
		weig[x] = module->data[x+1];
		acti[id+x] = module->acti_func;
	}
}

wmodule *wekuaLinear(wekuaContext *ctx, uint64_t input, uint64_t output, uint32_t deep, wacti *acti, uint8_t com){
	if (input*output*deep == 0){
		return NULL;
	}
	wmodule *linear_module = (wmodule*) calloc(1, sizeof(wmodule));
	linear_module->data = (void**) calloc(deep+1, sizeof(void*));
	linear_module->data[0] = (void*) calloc(1, 4);
	((uint32_t*)linear_module->data[0])[0] = deep;
	for (uint32_t x=1; x<deep; x++){
		linear_module->data[x] = (void*) wekuaMatrixRandUniform(ctx, input+1, input, -1.0, 0.0, 1.0, 0.0, com);
	}
	linear_module->data[deep] = (void*) wekuaMatrixRandUniform(ctx, input+1, output, -1.0, 0.0, 1.0, 0.0, com);
	linear_module->func = &runWekuaLinear;
	linear_module->free_func = &freeWekuaLinear;
	linear_module->set_cache_id = &linear_set_cache_id;
	if (acti != NULL){
		linear_module->acti_func = acti;
	}else{
		linear_module->acti_func = wekuaFLinear();
	}
	linear_module->arch_id = -1;
	linear_module->com = com;
	linear_module->nmod = deep;
	return linear_module;
}

wmatrix *runLinearNeuron(wmatrix *a, wmatrix *w, wacti *acti){
	wmatrix *in = wekuaMatrixResize(a, a->shape[0], a->shape[1]+1, 1.0, 0.0);
	wmatrix *output = wekuaMatrixProduct(in, w);
	runWekuaActi(acti, output);
	wekuaFreeMatrix(in);
	return output;
}

wmatrix *runWekuaLinear(void *m, wmatrix *input){
	wmodule *module = m;
	if (m == NULL || input == NULL){
		return NULL;
	}else if (input->shape[1]+1 != ((wmatrix*)module->data[1])->shape[0]){
		return NULL;
	}
	wmatrix *output, **tmp;
	uint8_t d = 1;
	tmp = (wmatrix**) calloc(2, sizeof(wmatrix*));
	if (module->pseq[0] == 0){
		((wmatrix**)module->cache)[0] = wekuaMatrixCopy(input);
		module->pseq[0]++;
	}
	tmp[0] = runLinearNeuron(input, (wmatrix*)module->data[1], module->acti_func);
	for (uint32_t x=2; x <= ((uint32_t*)module->data[0])[0]; x++){
		tmp[d] = runLinearNeuron(tmp[d^1], (wmatrix*)module->data[x], module->acti_func);
		d ^= 1;
		if (module->arch_id > 0){
			module->cache[module->pseq[0]] = tmp[d];
			module->w_id[module->pseq[0]-1] = module->arch_id+x-2;
			module->pseq[0]++;
		}else{
			wekuaFreeMatrix(tmp[d]);
		}
	}
	output = tmp[d^1];
	free(tmp);
	if (module->arch_id >= 0){
		module->cache[module->pseq[0]] = output;
		module->w_id[module->pseq[0]-1] = module->arch_id+((uint32_t*)module->data[0])[0]-1;
		module->pseq[0]++;
	}
	return output;
}

void freeWekuaLinear(void *m){
	wmodule *module = m;
	for (uint32_t x=1; x <= ((uint32_t*)module->data[0])[0]; x++){
		wekuaFreeMatrix(module->data[x]);
	}
	free(((uint32_t*)module->data[0]));
	free(module->data);
}
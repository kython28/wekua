#include "wekua.h"

void acti_func_none(wmatrix *a){
	return;
}

void linear_set_cache_id(void *m, int64_t id, void *cache, void *w, uint32_t *pseq, uint8_t *fa){
	((wmodule*)m)->arch_id = id;
	((wmodule*)m)->cache = cache;
	((wmodule*)m)->pseq = pseq;
	for (uint32_t x=0; x<((wmodule*)m)->nmod; x++){
		((wmatrix**)w)[x] = (wmatrix*) ((wmodule*)m)->data[x+1];
		if (((wmodule*)m)->acti_func == &wekuaSigmoid){
			fa[x] = 1;
		}else if (((wmodule*)m)->acti_func == &wekuaTanh){
			fa[x] = 2;
		}else if (((wmodule*)m)->acti_func == &wekuaReLU){
			fa[x] = 3;
		}else if (((wmodule*)m)->acti_func == &wekuaLeakyReLU){
			fa[x] = 4;
		}else if (((wmodule*)m)->acti_func == &wekuaSoftplus){
			fa[x] = 5;
		}
	}
}

wmodule *wekuaLinear(wekuaContext *ctx, uint32_t input, uint32_t output, uint32_t deep, void (*acti_func)(wmatrix *a), uint8_t com){
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
	if (acti_func != NULL){
		linear_module->acti_func = acti_func;
	}else{
		linear_module->acti_func = &acti_func_none;
	}
	linear_module->arch_id = -1;
	linear_module->com = com;
	linear_module->nmod = deep;
	return linear_module;
}

wmatrix *runLinearNeuron(wmatrix *a, wmatrix *w, void (*acti_func)(wmatrix *a)){
	wmatrix *in = wekuaMatrixResize(a, a->r, a->c+1, 1.0, 0.0);
	wmatrix *output = wekuaMatrixProduct(in, w);
	acti_func(output);
	wekuaFreeMatrix(in);
	return output;
}

wmatrix *runWekuaLinear(void *m, wmatrix *input){
	wmodule *module = m;
	if (m == NULL || input == NULL){
		return NULL;
	}else if (input->c+1 != ((wmatrix*)module->data[1])->r){
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
			((wmatrix**)module->cache)[module->pseq[0]] = tmp[d];
			module->pseq[0]++;
		}else{
			wekuaFreeMatrix(tmp[d]);
		}
	}
	output = tmp[d^1];
	free(tmp);
	if (module->arch_id > 0){
		((wmatrix**)module->cache)[module->pseq[0]] = output;
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
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
	// wmatrix *in = wekuaMatrixResize(a, a->shape[0], a->shape[1]+1, 1.0, 0.0);
	//wmatrix *output = wekuaMatrixProduct(a, w);
	//runWekuaActi(acti, output);
	//wekuaFreeMatrix(in);
	//return output;
	cl_event e;
	wmatrix *in, *output;
	uint64_t row = a->shape[0], col = w->shape[1];
	output = wekuaFillMatrix(a->ctx, row, col+1, 1.0, 0.0);
	in = wekuaCutMatrix(output, 0, col, 0, row);
	wekuaBlasGemm(1.0, 0.0, 0, a, 0, w, 0.0, 0.0, in, 0, NULL, &e);
	runWekuaActi(acti, in, 1, &e);
	wekuaFreeMatrix(in, 0, NULL);
	clReleaseEvent(e);
	return output;
}

wmatrix *runWekuaLinear(void *m, wmatrix *input, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	wmodule *module = m;
	if (m == NULL || input == NULL){
		return NULL;
	}else if (input->shape[1]+1 != ((wmatrix*)module->data[1])->shape[0]){
		return NULL;
	}
	wmatrix *output, *in, **tmp, **cache, **wei;
	uint8_t d = 1;
	uint32_t *pseq = module->pseq, n_wei = ((uint32_t*)module->data[0])[0];
	int64_t arch_id, *w_id;
	w_id = module->w_id;
	arch_id = module->arch_id;
	cache = module->cache;
	wei = (wmatrix**) &module->data[1];
	cl_event e;

	tmp = (wmatrix**) calloc(2, sizeof(wmatrix*));
	if (pseq[0] == 0 && module->arch_id >= 0){
		//((wmatrix**)module->cache)[0] = wekuaMatrixCopy(input);
		in = wekuaMatrixResize(input, input->shape[0], input->shape[1]+1, 1.0, 0.0, 0, NULL, &e);
		cache[0] = in;
		pseq[0]++;
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}else if (input->parent != NULL && module->arch_id >= 0){
		in = cache[pseq[0]-1];
		if (input->parent != in){
			in = wekuaMatrixResize(input, input->shape[0], input->shape[1]+1, 1.0, 0.0, 0, NULL, &e);
			wekuaMatrixPrint(in, 0, NULL);
			clWaitForEvents(1, &e);
			clReleaseEvent(e);
		}
	}else{
		in = wekuaMatrixResize(input, input->shape[0], input->shape[1]+1, 1.0, 0.0, 0, NULL, &e);
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}

	tmp[0] = runLinearNeuron(in, wei[0], module->acti_func);
	for (uint32_t x=1; x < n_wei; x++){
		tmp[d] = runLinearNeuron(tmp[d^1], wei[x], module->acti_func);
		d ^= 1;
		if (module->arch_id >= 0){
			cache[pseq[0]] = tmp[d^1];
			w_id[pseq[0]-1] = arch_id+x-2;
			pseq[0]++;
		}else{
			wekuaFreeMatrix(tmp[d], 0, NULL);
		}
	}
	// output = tmp[d^1];
	if (arch_id >= 0){
		output = wekuaCutMatrix(tmp[d^1], 0, tmp[d^1]->shape[1]-1, 0, tmp[d^1]->shape[0]);
		cache[pseq[0]] = tmp[d^1];
		w_id[pseq[0]-1] = arch_id+n_wei-1;
		pseq[0]++;
	}else{
		output = wekuaMatrixResize(tmp[d^1], tmp[d^1]->shape[0], tmp[d^1]->shape[1]-1, 0.0, 0.0, 0, NULL, &e);
		wekuaFreeMatrix(tmp[d^1], 1, &e);
		clReleaseEvent(e);
	}
	free(tmp);
	return output;
}

void freeWekuaLinear(void *m, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	wmodule *module = m;
	for (uint32_t x=1; x <= ((uint32_t*)module->data[0])[0]; x++){
		wekuaFreeMatrix(module->data[x], 0, NULL);
	}
	free(((uint32_t*)module->data[0]));
	free(module->data);
}
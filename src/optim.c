#include "wekua.h"

// Gradient Descent

int step_optim_gd(void *data, void *optim_data, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weigth, wmatrix bias){
	void *lr = data;
	void *lri = &((uint8_t*)data)[dl];
	uint32_t evn = 0;
	cl_event e[2];
	int ret;

	ret = wekuaBlasAxpy(error, weigth, lr, lri, 0, NULL, e);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;
	evn++;

	if (error_b != NULL){
		ret = wekuaBlasAxpy(error_b, bias, lr, lri, 0, NULL, &e[1]);
		if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;
		evn++;
	}

	wk_step_optim_gd_fail:
	clWaitForEvents(evn, e);
	for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);

	return ret;
}

void free_optim_gd(void *opti, uint32_t nw, cl_event *be){
	if (opti == NULL) return;
	clWaitForEvents(nw, be);
	woptim optim = opti;
	free(optim->data);
	free(opti);
}

woptim wekuaOptimGD(wekuaContext ctx, wnetwork net, void *lr, void *lri, uint8_t dtype){
	if ((lr == NULL && lri == NULL) || net == NULL || ctx == NULL) return NULL;

	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	void *data = calloc(2, ctx->dtype_length[dtype]);
	
	opti->net = net;
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->data = data;
	opti->step = &step_optim_gd;
	opti->free_func = &free_optim_gd;

	if (dtype == WEKUA_DTYPE_DOUBLE){
		((double*)data)[0] = -1.0 * ((double*)lr)[0];
		if (lri != NULL){
			((double*)data)[1] = -1.0 * ((double*)lri)[0];
		}
	}else{
		((float*)data)[0] = -1.0 * ((float*)lr)[0];
		if (lri != NULL){
			((float*)data)[1] = -1.0 * ((float*)lri)[0];
		}
	}

	return opti;
}

int wekuaOptimStep(woptim optim, werror *error, wcache *cache){
	if (optim == NULL || error == NULL || cache == NULL) return CL_INVALID_ARG_VALUE;

	int ret;
	wnetwork net = optim->net;
	wneuron *neurons = net->neurons;
	uint32_t nneur = net->nneur;
	uint32_t x = 0;
	void *opti_data = optim->data;
	int (*func)(void *, void*, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix);

	func = optim->step;

	wneuron neuron_tmp;
	wcache cache_tmp;
	werror error_tmp;

	for (; x<nneur; x++){
		neuron_tmp = neurons[x];
		cache_tmp = cache[x];
		error_tmp = error[nneur - x - 1];

		ret = neuron_tmp->step(neuron_tmp, opti_data, error_tmp, cache_tmp, func);
		if (ret != CL_SUCCESS) break;
	}

	return ret;
}
#include "wekua.h"

// Gradient Descent

int step_optim_gd(void *data, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weigth, wmatrix bias){
	void *lr = data;
	void *lri = &((uint8_t*)data)[dl];
	uint32_t evn = 0;
	cl_event e[4];
	int ret;

	ret = wekuaBlasScalar(error, lr, lri, 0, NULL, e);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;
	evn++;

	ret = wekuaMatrixSub(weigth, error, 1, e, &e[1]);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;
	evn++;

	ret = wekuaBlasScalar(error_b, lr, lri, 0, NULL, &e[2]);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;
	evn++;

	ret = wekuaMatrixSub(bias, error_b, 1, &e[2], &e[3]);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;
	evn++;

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

woptim wekuaOptimGD(wekuaContext ctx, void *lr, void *lri, uint8_t dtype){
	if (lr == NULL && lri == NULL && net == NULL) return NULL;

	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	uint32_t dl = ctx->dtype_length[dtype];
	void *data = calloc(2, dl);
	
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->data = data;
	opti->step = &step_optim_gd;
	opti->free_func = &free_optim_gd;
	
	memcpy(data, lr, dl);
	if (lri != NULL){
		memcpy(((uint8_t*)data)[dl], lri, dl);
	}

	return opti;
}

woptim wekuaOptimGDM(wekuaContext ctx, void *alpha, void *alphai, void *beta, void *betai, uint8_t dtype); // Gradient descent momentum optimization
woptim wekuaOptimAdaGrad(wekuaContext ctx, void *lr, void *lri, uint8_t dtype); // Gradient adaptive optimization
woptim wekuaOptimRMSProp(wekuaContext ctx, void *alpha, void *alphai, void *beta, void *betai, uint8_t dtype); // RMSProp optimization
woptim wekuaOptimAdaDelta(wekuaContext ctx, void *lr, void *lri, uint8_t dtype); // AdaDelta optimization

int wekuaOptimStep(woptim optim, wnetwork net, werror *error, wcache *cache){
	if (optim == NULL || net == NULL || error == NULL || wcache == NULL) return CL_INVALID_ARG_VALUE;

	int ret;
	wneuron neurons = net->neurons;
	uint32_t nneur = net->nneur;
	uint32_t x = 0;
	void *opti_data = optim->data;
	int (*func)(void *, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix);

	func = &optim->step;

	for (; x<nneur; x++){
		register wneuron neuron_tmp = neurons[x];
		register wcache cache_tmp = cache[x];
		register werror error_tmp = error[nneur - x - 1];

		ret = neuron_tmp->step(neuron_tmp, opti_data, error_tmp, cache_tmp, func);
		if (ret != CL_SUCCESS) break;

		neuron_tmp->free_cache(cache_tmp);
		neuron_tmp->free_error(error_tmp);
	}

	if (ret == CL_SUCCESS){
		free(error);
		free(cache);
	}

	return ret;
}
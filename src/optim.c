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

	ret = wekuaMatrixSub(weigth, error, 1, e, &e[1]);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;

	ret = wekuaBlasScalar(error_b, lr, lri, 0, NULL, &e[2]);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;

	ret = wekuaMatrixSub(bias, error_b, 1, &e[2], &e[3]);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;

	wk_step_optim_gd_fail:
	clWaitForEvents(evn, e);
	for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);

	return ret;
}

woptim wekuaOptimGD(void *lr, void *lri, wnetwork net){
	if (lr == NULL && lri == NULL && net == NULL) return NULL;

	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	wekuaContext ctx = net->ctx;
	uint32_t dl = ctx->dtype_length[net->dtype];
	void *data = calloc(2, dl);
	
	opti->net = net;
	opti->data = data;
	
	memcpy(data, lr, dl);
	if (lri != NULL){
		memcpy(((uint8_t*)data)[dl], lri, dl);
	}
	

}


woptim wekuaOptimGDM(void *alpha, void *alphai, void *beta, void *betai, wnetwork net); // Gradient descent momentum optimization
woptim wekuaOptimAdaGrad(void *lr, void *lri, wnetwork net); // Gradient adaptive optimization
woptim wekuaOptimRMSProp(void *alpha, void *alphai, void *beta, void *betai, wnetwork net); // RMSProp optimization
woptim wekuaOptimAdaDelta(void *lr, void *lri, wnetwork net); // AdaDelta optimization

int wekuaOptimStep(woptim optim, werror *error, wcache *cache);
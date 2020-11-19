#include "wekua.h"

void *get_one(uint8_t dtype, uint32_t dl);

int run_alinear(void *data, wmatrix input, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	return CL_SUCCESS;
}

wmatrix get_dev(void *data, wmatrix input){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;

	void *one = get_one(dtype, ctx->dtype_length[dtype]);
	if (one == NULL) return NULL;

	wmatrix dev = wekuaFillMatrix(ctx, input->shape[0], input->shape[1], one, NULL, dtype);
	free(one);
	return dev;
}

void free_linear(void *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	free(a);
}

wacti wekuaActiLinear(){
	wacti acti = (wacti) calloc(1, sizeof(struct _w_acti));
	if (acti == NULL) return NULL;
	
	acti->run_acti = &run_alinear;
	acti->get_dev = &get_dev;
	acti->free_func = &free_linear;

	return acti;
}
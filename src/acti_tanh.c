#include "wekua.h"

void *get_one(uint8_t dtype, uint32_t dl);

int run_tanh(void *data, wmatrix input, uint32_t nw, cl_event *be){
	cl_event e;

	int ret = wekuaMatrixTanh(input, nw, be, &e);
	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}

	return ret;
}

wmatrix get_dev_tanh(void *data, wmatrix input){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	int ret;
	cl_event e[3];

	void *one = get_one(dtype, ctx->dtype_length[dtype]);
	if (one == NULL) return NULL;

	wmatrix dev = wekuaFillMatrix(ctx, input->shape[0], input->shape[1], one, NULL, dtype);
	free(one);
	if (dev == NULL) return NULL;

	wmatrix b = wekuaMatrixCopy(input, 0, NULL, e);
	if (b == NULL){
		wekuaFreeMatrix(dev, 0, NULL);
		return NULL;
	}

	ret = wekuaMatrixDot(b, b, 1, e, &e[1]);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(b, 1, e);
		wekuaFreeMatrix(dev, 0, NULL);
		clReleaseEvent(e[0]);
		return NULL;
	}
     
	ret = wekuaMatrixSub(dev, b, 1, &e[1], &e[2]);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(dev, 1, &e[1]);
		wekuaFreeMatrix(b, 0, NULL);
		clReleaseEvent(e[0]);
		clReleaseEvent(e[1]);
		return NULL;
	}

	wekuaFreeMatrix(b, 1, &e[2]);
	clReleaseEvent(e[0]);
	clReleaseEvent(e[1]);
	clReleaseEvent(e[2]);

	return dev;
}

void free_tanh(void *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	free(a);
}

wacti wekuaActiSigmoid(){
	wacti acti = (wacti) calloc(1, sizeof(struct _w_acti));
	if (acti == NULL) return NULL;
	
	acti->run_acti = &run_tanh;
	acti->get_dev = &get_dev_tanh;
	acti->free_func = &free_tanh;

	return acti;
}
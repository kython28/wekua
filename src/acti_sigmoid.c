#include "wekua.h"

void *get_one(uint8_t dtype, uint32_t dl);

int run_sigmoid(void *data, wmatrix input, uint32_t nw, cl_event *be){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	cl_kernel kernel;
	cl_event e;

	if (compileKernel(ctx, WEKUA_KERNEL_SIGMOID, dtype)) return CL_BUILD_PROGRAM_FAILURE;

	kernel = ctx->kernels[WEKUA_KERNEL_SIGMOID*10 + dtype];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->imag);
	clSetKernelArg(kernel, 2, 1, &input->com);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &input->vl_shape[2], &input->work_items[8], nw, be, &e);

	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}
	
	return ret;
}

wmatrix get_dev_sig(void *data, wmatrix input){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	int ret;
	cl_event e[2];

	void *one = get_one(dtype, ctx->dtype_length[dtype]);
	if (one == NULL) return NULL;

	wmatrix dev = wekuaFillMatrix(ctx, input->shape[0], input->shape[1], one, NULL, dtype);
	free(one);
	if (dev == NULL) return NULL;
     
	ret = wekuaMatrixSub(dev, input, 0, NULL, e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(dev, 0, NULL);
		return NULL;
	}

	ret = wekuaMatrixDot(dev, input, 1, e, &e[1]);
	if (ret != CL_SUCCESS){
		clWaitForEvents(1, e);
		clReleaseEvent(e[0]);
		wekuaFreeMatrix(dev, 0, NULL);
		return NULL;
	}

	clWaitForEvents(2, e);
	clReleaseEvent(e[0]);
	clReleaseEvent(e[1]);

	return dev;
}

void free_sigmoid(void *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	free(a);
}

wacti wekuaActiTanh(){
	wacti acti = (wacti) calloc(1, sizeof(struct _w_acti));
	if (acti == NULL) return NULL;
	
	acti->run_acti = &run_sigmoid;
	acti->get_dev = &get_dev_sig;
	acti->free_func = &free_sigmoid;

	return acti;
}
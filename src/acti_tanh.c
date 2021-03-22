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
	uint8_t dtype = input->dtype, com = input->com;
	cl_kernel kernel;
	cl_event e;
	int ret;

	wmatrix dev;

	if (compileKernel(ctx, WEKUA_KERNEL_TANH_DEV, dtype)) return NULL;

	kernel = ctx->kernels[WEKUA_KERNEL_TANH_DEV*10 + dtype];

	if (com){
		dev = wekuaAllocComplexMatrix(ctx, input->shape[0], input->shape[1], dtype);
	}else{
		dev = wekuaAllocMatrix(ctx, input->shape[0], input->shape[1], dtype);
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev->imag);
	clSetKernelArg(kernel, 4, 1, &com);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &input->vl_shape[2], &input->work_items[8], 0, NULL, &e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(dev, 0, NULL);
		return NULL;
	}

	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	return dev;
}

void free_tanh(void *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	free(a);
}

wacti wekuaActiTanh(){
	wacti acti = (wacti) calloc(1, sizeof(struct _w_acti));
	if (acti == NULL) return NULL;
	
	acti->run_acti = &run_tanh;
	acti->get_dev = &get_dev_tanh;
	acti->free_func = &free_tanh;

	return acti;
}
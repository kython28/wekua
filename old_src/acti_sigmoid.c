#include "../headers/acti.h"

void *get_one(uint8_t dtype, uint32_t dl);

int run_sigmoid(__attribute__((unused)) void *data, __attribute__((unused)) wmatrix input, uint32_t nw, cl_event *be){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_SIGMOID, dtype, input->com);
	cl_event e;

	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->imag);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &input->vl_shape[2], &input->work_items[8], nw, be, &e);

	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}
	
	return ret;
}

wmatrix get_dev_sig(__attribute__ ((unused)) void *data, wmatrix input){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype, com = input->com;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_SIGMOID_DEV, dtype, com);
	cl_event e;
	int ret;

	wmatrix dev;

	if (kernel == NULL) return NULL;

	if (com){
		dev = wekuaAllocComplexMatrix(ctx, input->shape[0], input->shape[1], dtype);
	}else{
		dev = wekuaAllocMatrix(ctx, input->shape[0], input->shape[1], dtype);
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev->imag);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &input->vl_shape[2], &input->work_items[8], 0, NULL, &e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(dev, 0, NULL);
		return NULL;
	}

	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	return dev;
}

void free_sigmoid(void *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	free(a);
}

wacti wekuaActiSigmoid(void){
	wacti acti = (wacti) calloc(1, sizeof(struct _w_acti));
	if (acti == NULL) return NULL;
	
	acti->run_acti = &run_sigmoid;
	acti->get_dev = &get_dev_sig;
	acti->free_func = &free_sigmoid;

	return acti;
}

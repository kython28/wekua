#include "wekua.h"

int run_relu(void *data, wmatrix input, uint32_t nw, cl_event *be){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	cl_kernel kernel;
	cl_event e;

	if (compileKernel(ctx, WEKUA_KERNEL_RELU, dtype)) return CL_BUILD_PROGRAM_FAILURE;

	kernel = ctx->kernels[WEKUA_KERNEL_RELU*10 + dtype];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->imag);
	clSetKernelArg(kernel, 2, 8, &input->col);
	clSetKernelArg(kernel, 3, 1, &input->com);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, input->shape, &input->work_items[4], nw, be, &e);

	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}
	
	return ret;
}

wmatrix get_dev_relu(void *data, wmatrix input){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	cl_kernel kernel;
	cl_event e;

	if (compileKernel(ctx, WEKUA_KERNEL_RELU_DEV, dtype)) return NULL;

	wmatrix dev = wekuaAllocMatrix(ctx, input->shape[0], input->shape[1], dtype);
	if (input->com){
		if (createComplexMatrix(dev)){
			wekuaFreeMatrix(dev, 0, NULL);
			return NULL;
		}
	}

	kernel = ctx->kernels[WEKUA_KERNEL_RELU_DEV*10 + dtype];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev->real);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev->col);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, input->shape, &input->work_items[4], 0, NULL, &e);

	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}else{
		wekuaFreeMatrix(dev, 0, NULL);
		dev = NULL;
	}
	return dev;
}

void free_relu(void *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	free(a);
}

wacti wekuaActiReLU(){
	wacti acti = (wacti) calloc(1, sizeof(struct _w_acti));
	if (acti == NULL) return NULL;
	
	acti->run_acti = &run_relu;
	acti->get_dev = &get_dev_relu;
	acti->free_func = &free_relu;

	return acti;
}
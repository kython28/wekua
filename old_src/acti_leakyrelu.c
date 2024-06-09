#include "../headers/acti.h"

int run_lrelu(void *data, wmatrix input, uint32_t nw, cl_event *be){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	uint32_t dl = ctx->dtype_length[dtype];
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_LEAKY_RELU, dtype, input->com);
	cl_event e;

	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input->imag);
	clSetKernelArg(kernel, 2, dl, data);
	clSetKernelArg(kernel, 3, dl, data + dl);
	clSetKernelArg(kernel, 4, 8, &input->col);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, input->shape, &input->work_items[4], nw, be, &e);

	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}
	
	return ret;
}

wmatrix get_dev_lrelu(void *data, wmatrix input){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	uint32_t dl = ctx->dtype_length[dtype];
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_LEAKY_RELU_DEV, dtype, input->com);
	cl_event e;

	if (kernel == NULL) return NULL;

	wmatrix dev = wekuaAllocMatrix(ctx, input->shape[0], input->shape[1], dtype);
	if (input->com){
		if (createComplexMatrix(dev)){
			wekuaFreeMatrix(dev, 0, NULL);
			return NULL;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev->real);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev->imag);
	clSetKernelArg(kernel, 3, dl, data);
	clSetKernelArg(kernel, 4, dl, data + dl);
	clSetKernelArg(kernel, 5, 8, &input->col);

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

void free_lrelu(void *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	wacti b = a;
	free(b->data);
	free(a);
}

wacti wekuaActiLeakyReLU(wekuaContext ctx, void *alpha, void *alphai, uint8_t dtype){
	wacti acti = (wacti) calloc(1, sizeof(struct _w_acti));
	if (acti == NULL) return NULL;

	uint32_t dl = ctx->dtype_length[dtype];

	void *data = calloc(2, dl);
	memcpy(data, alpha, dl);
	if (alphai != NULL) memcpy(data+dl, alphai, dl);
	
	acti->data = data;
	acti->run_acti = &run_lrelu;
	acti->get_dev = &get_dev_lrelu;
	acti->free_func = &free_lrelu;

	return acti;
}
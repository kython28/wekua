#include "../headers/acti.h"

wmatrix get_dev_sig(void *data, wmatrix input);
void *get_one(uint8_t dtype, uint32_t dl);

int run_softmax(__attribute__((unused)) void *data, wmatrix input, uint32_t nw, cl_event *be){
	wekuaContext ctx = input->ctx;
	uint8_t dtype = input->dtype;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_SOFTMAX, dtype, input->com);
	cl_event e;

	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input->real);
	clSetKernelArg(kernel, 1, sizeof(uint64_t), &input->shape[1]);
	clSetKernelArg(kernel, 2, sizeof(uint64_t), &input->col);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, input->shape, &input->work_items[6], nw, be, &e);

	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}
	
	return ret;
}

void free_softmax(void *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	free(a);
}

wacti wekuaActiSoftmax(){
	wacti acti = (wacti) calloc(1, sizeof(struct _w_acti));
	if (acti == NULL) return NULL;
	
	acti->run_acti = &run_softmax;
	acti->get_dev = &get_dev_sig;
	acti->free_func = &free_softmax;

	return acti;
}

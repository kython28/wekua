#include "regularization.h"

static int sum_weight(wmatrix a, wmatrix b, void *alpha, void *beta, uint32_t *nevents, cl_event *e){
	wekuaContext ctx = a->ctx;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_SUM, a->dtype, a->com);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 8, &a->vl_shape[1]);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &b->shape[1], &b->work_items[7], 1, &e[1], &e[2]);
	if (ret != CL_SUCCESS) goto sum_weight_error;
	nevents[0]++;

	ret = wekuaBlasScalar(b, alpha, beta, 1, &e[2], &e[3]);
	if (ret != CL_SUCCESS) goto sum_weight_error;
	nevents[0]++;

	sum_weight_error:
	return ret;
}

wmatrix wekuaL1Regularization(wmatrix weight, void *alpha, void *beta){
	cl_event e[4];
	wmatrix a = wekuaMatrixCopy(weight, 0, NULL, e);
	

	uint8_t dtype = weight->dtype;
	uint8_t com = weight->com;
	wekuaContext ctx = weight->ctx;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_L1_REGULARIZATION, dtype, com);
	uint32_t nevents = 1;

	wmatrix b = NULL;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->vl_shape[1]);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->vl_shape, a->work_items, 1, e, &e[1]);
	if (ret != CL_SUCCESS) goto wekuaL1Regularization_error;
	nevents++;

	b = wekuaAllocMatrix(ctx, 1, a->shape[0], dtype);
	ret = sum_weight(a, b, alpha, beta, &nevents, e);

	wekuaL1Regularization_error:
	if (nevents){
		clWaitForEvents(nevents, e);
		for (uint32_t i = 0; i<nevents; i++) clReleaseEvent(e[i]);
	}

	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(b, 0, NULL);
		b = NULL;
	}

	wekuaFreeMatrix(a, 0, NULL);
	return b;
}

wmatrix wekuaL2Regularization(wmatrix weight, void *alpha, void *beta){
	cl_event e[4];
	wmatrix a = wekuaMatrixCopy(weight, 0, NULL, &e[1]);

	wekuaContext ctx = weight->ctx;
	uint8_t dtype = weight->dtype;
	uint32_t nevents = 1;
	wmatrix b = NULL;

	double d_alpha_ = 0.0;
	double d_beta_ = 0.0;
	float f_alpha_ = 0.0;
	float f_beta_ = 0.0;

	if (dtype == WEKUA_DTYPE_DOUBLE){
		if (alpha) {
			d_alpha_ = 2.0 * ((double*)alpha)[0];
			alpha = &d_alpha_;
		}
		if (beta) {
			d_beta_ = 2.0 * ((double*)beta)[0];
			beta = &d_beta_;
		}
	}else{
		if (alpha){
			f_alpha_ = 2.0f * ((float*)alpha)[0];
			alpha = &f_alpha_;
		}
		if (beta) {
			f_beta_ = 2.0f * ((float*)beta)[0];
			beta = &f_beta_;
		}
	}

	b = wekuaAllocMatrix(ctx, 1, a->shape[0], dtype);
	int ret = sum_weight(a, b, alpha, beta, &nevents, e);

	if (nevents){
		clWaitForEvents(nevents, &e[1]);
		for (uint32_t i = 1; i<nevents; i++) clReleaseEvent(e[i]);
	}

	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(b, 0, NULL);
		b = NULL;
	}

	wekuaFreeMatrix(a, 0, NULL);
	return b;
}

int wekuaAddRegularization(wmatrix regularization, wmatrix dev_error, uint32_t nw, cl_event *be, cl_event *e){
	wekuaContext ctx = regularization->ctx;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_REGULARIZATION, regularization->dtype, regularization->com);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_error->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_error->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &regularization->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &regularization->imag);
	clSetKernelArg(kernel, 4, 8, &dev_error->vl_shape[1]);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, dev_error->vl_shape, dev_error->work_items, nw, be, e);
}

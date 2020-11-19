#include "wekua.h"

const uint64_t zero_blas = 0;

void getLWI(uint64_t *x, uint64_t *y, uint32_t si, uint64_t max);

int wekuaBlasAxpy(wmatrix x, wmatrix y, void *alpha, void *beta, uint32_t nw, cl_event *be, cl_event *e){
	if (x == NULL || y == NULL){
		return CL_INVALID_MEM_OBJECT;
	}else if (x->dtype != y->dtype){
		return CL_INVALID_MEM_OBJECT;
	}else if (memcmp(x->vl_shape, y->vl_shape, 16) != 0){
		return CL_INVALID_MEM_OBJECT;
	}
	uint8_t dtype = x->dtype, com = x->com|y->com;
	uint32_t len;

	wekuaContext ctx = x->ctx;
	if (compileKernel(ctx, WEKUA_KERNEL_AXPY, dtype)){
		return CL_COMPILE_PROGRAM_FAILURE;
	}
	len = ctx->dtype_length[dtype];

	if (com){
		if (createComplexMatrix(x) || createComplexMatrix(y)){
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
	}

	cl_kernel kernel = ctx->kernels[WEKUA_KERNEL_AXPY*10+dtype];
	if (alpha == NULL) alpha = ((uint64_t*)&zero_blas);
	if (beta == NULL) beta = ((uint64_t*)&zero_blas);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &x->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &y->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &y->imag);

	clSetKernelArg(kernel, 4, len, alpha);
	clSetKernelArg(kernel, 5, len, beta);
	clSetKernelArg(kernel, 6, 8, &x->vl_shape[1]);
	clSetKernelArg(kernel, 7, 1, &com);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, x->vl_shape, x->work_items, nw, be, e);
}

int wekuaBlasScalar(wmatrix x, void *alpha, void *beta, uint32_t nw, cl_event *be, cl_event *e){
	if (x == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	wekuaContext ctx = x->ctx;
	uint8_t dtype = x->dtype;
	uint32_t len = ctx->dtype_length[dtype];

	if (alpha == NULL) alpha = (uint64_t*)&zero_blas;
	if (beta == NULL) beta = (uint64_t*)&zero_blas;

	if (memcmp(beta, &zero_blas, len) != 0){
		if (createComplexMatrix(x)){
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
	}

	if (compileKernel(ctx, WEKUA_KERNEL_SCAL, dtype)){
		return CL_COMPILE_PROGRAM_FAILURE;
	}

	cl_kernel kernel = ctx->kernels[WEKUA_KERNEL_SCAL*10+dtype];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &x->imag);
	clSetKernelArg(kernel, 2, len, alpha);
	clSetKernelArg(kernel, 3, len, beta);
	clSetKernelArg(kernel, 4, 1, &x->com);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &x->vl_shape[2], &x->work_items[8], nw, be, e);
}

int wekuaBlasFastGemm(
	void *ralpha, void *ialpha, uint8_t a_trans, wmatrix a, uint8_t b_trans, wmatrix b,
	void *rbeta, void *ibeta, wmatrix c, uint32_t nw, cl_event *be
);

int wekuaBlasGemm(void *ralpha, void *ialpha, uint8_t a_trans, wmatrix a, uint8_t b_trans, wmatrix b,
	void *rbeta, void *ibeta, wmatrix c, uint32_t nw, cl_event *be
){
	if (a == NULL || b == NULL || c == NULL){
		return CL_INVALID_MEM_OBJECT;
	}else if ((a->dtype != b->dtype) || (b->dtype != c->dtype)) return CL_INVALID_MEM_OBJECT;
	else if (ralpha == NULL && ialpha == NULL && rbeta == NULL && ibeta == NULL) return CL_INVALID_ARG_VALUE;

	if ((a->com|b->com|c->com) || ibeta != NULL || ialpha != NULL){
		if (createComplexMatrix(a) || createComplexMatrix(b) || createComplexMatrix(c)){
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
	}

	if (c->shape[0] == c->shape[1] && c->shape[0]*c->shape[0]*c->shape[0] >= 27000000){
		return wekuaBlasFastGemm(ralpha, ialpha, a_trans, a, b_trans, b, rbeta, ibeta, c, nw, be);
	}

	wekuaContext ctx = a->ctx;
	uint8_t dtype = a->dtype;
	uint32_t len = ctx->dtype_length[dtype];

	cl_kernel kernel;
	cl_event e;
	wmatrix x, y;

	if (compileKernel(ctx, WEKUA_KERNEL_GEMM, dtype)){
		return CL_COMPILE_PROGRAM_FAILURE;
	}
	kernel = ctx->kernels[WEKUA_KERNEL_GEMM*10+dtype];

	if (b_trans == 0){
		y = wekuaMatrixTrans(b, nw, be, &e);
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}else{
		y = b;
	}

	if (a_trans){
		x = wekuaMatrixTrans(a, nw, be, &e);
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}else{
		x = a;
	}

	if (ralpha == NULL) ralpha = ((uint64_t*)&zero_blas);
	if (ialpha == NULL) ialpha = ((uint64_t*)&zero_blas);
	if (rbeta == NULL) rbeta = ((uint64_t*)&zero_blas);
	if (ibeta == NULL) ibeta = ((uint64_t*)&zero_blas);

	// Matrix
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &x->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &y->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &y->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &c->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &c->imag);

	// Scalars
	clSetKernelArg(kernel, 6, len, ralpha);
	clSetKernelArg(kernel, 7, len, ialpha);
	clSetKernelArg(kernel, 8, len, rbeta);
	clSetKernelArg(kernel, 9, len, ibeta);

	// Dimensions
	clSetKernelArg(kernel, 10, 8, &x->vl_shape[1]);
	clSetKernelArg(kernel, 11, 8, &y->vl_shape[1]);
	clSetKernelArg(kernel, 12, 8, &c->col); // C Matrix dimension

	// Does the matrix use complex numbers
	clSetKernelArg(kernel, 13, 1, &a->com);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, c->shape, &c->work_items[4], nw, be, &e);

	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		if (x != a) wekuaFreeMatrix(x, 0, NULL);
		if (y != b) wekuaFreeMatrix(y, 0, NULL);

		clReleaseEvent(e);
	}

	return ret;
}

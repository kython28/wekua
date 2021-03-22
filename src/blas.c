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
	clSetKernelArg(kernel, 6, 1, &com);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &x->vl_shape[2], &x->work_items[8], nw, be, e);
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

int wekuaBlasGemm(void *ralpha, void *ialpha, uint8_t a_trans, wmatrix a, uint8_t b_trans, wmatrix b,
	void *rbeta, void *ibeta, wmatrix c, uint32_t nw, cl_event *be
){
	if (ralpha == NULL && ialpha == NULL && rbeta == NULL && ibeta == NULL) return CL_INVALID_ARG_VALUE;
	else if (a == NULL || b == NULL || c == NULL) return CL_INVALID_MEM_OBJECT;
	else if ((a->dtype&b->dtype) != c->dtype) return CL_INVALID_MEM_OBJECT;

	if (ralpha == NULL) ralpha = (uint64_t*)&zero_blas;
	if (ialpha == NULL) ialpha = (uint64_t*)&zero_blas;
	if (rbeta == NULL) rbeta = (uint64_t*)&zero_blas;
	if (ibeta == NULL) ibeta = (uint64_t*)&zero_blas;

	wekuaContext ctx = a->ctx;
	cl_kernel kernel;
	cl_event e;

	wmatrix x = NULL, y = NULL;
	int ret;
	uint8_t dtype = a->dtype, com = a->com|b->com|c->com;
	uint32_t dl = ctx->dtype_length[dtype];

	if (compileKernel(ctx, WEKUA_KERNEL_GEMM, dtype)){
		return CL_BUILD_PROGRAM_FAILURE;
	}

	kernel = ctx->kernels[WEKUA_KERNEL_GEMM*10+dtype];

	clWaitForEvents(nw, be);

	if (com){
		if (createComplexMatrix(a)|createComplexMatrix(b)|createComplexMatrix(c)){
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
	}

	if (a_trans){
		x = wekuaMatrixTrans(a, 0, NULL, &e);
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}else x = a;

	if (b_trans) y = b;
	else{
		y = wekuaMatrixTrans(b, 0, NULL, &e);
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}

	if (x == NULL || y == NULL){
		ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;
		goto wekua_gemm_finish;
	}


	uint64_t shape[2], wi[2];

	memcpy(shape, c->shape, 16);

	if (shape[0]%2 != 0) shape[0]++;
	if (shape[1]%2 != 0) shape[1]++;

	shape[0] >>= 1;
	shape[1] >>= 1;

	getLWI(shape, wi, 2, ctx->max_work_group_size);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &x->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &y->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &y->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &c->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &c->imag);

	clSetKernelArg(kernel, 6, dl, ralpha);
	clSetKernelArg(kernel, 7, dl, ialpha);
	clSetKernelArg(kernel, 8, dl, rbeta);
	clSetKernelArg(kernel, 9, dl, ibeta);

	clSetKernelArg(kernel, 10, 8, &c->col);
	clSetKernelArg(kernel, 11, 8, &x->vl_shape[1]);
	clSetKernelArg(kernel, 12, 1, &c->com);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, shape, wi, 0, NULL, &e);

	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}

	wekua_gemm_finish:
	if (x != a) wekuaFreeMatrix(x, 0, NULL);
	if (y != b) wekuaFreeMatrix(y, 0, NULL);

	return ret;
}
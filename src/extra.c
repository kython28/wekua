#include "wekua.h"

uint64_t zero_extra = 0;

uint8_t isgreat(void *a, void *b, uint8_t dtype){
	if (dtype == WEKUA_DTYPE_INT8){
		if (((int8_t*)a)[0] > ((int8_t*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_UINT8){
		if (((uint8_t*)a)[0] > ((uint8_t*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_INT16){
		if (((int16_t*)a)[0] > ((int16_t*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_UINT16){
		if (((uint16_t*)a)[0] > ((uint16_t*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_INT32){
		if (((int32_t*)a)[0] > ((int32_t*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_UINT32){
		if (((uint32_t*)a)[0] > ((uint32_t*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_INT64){
		if (((int64_t*)a)[0] > ((int64_t*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_UINT64){
		if (((uint64_t*)a)[0] > ((uint64_t*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_FLOAT){
		if (((float*)a)[0] > ((float*)b)[0]) return 1;
		return 0;
	}
	else if (dtype == WEKUA_DTYPE_DOUBLE){
		if (((double*)a)[0] > ((double*)b)[0]) return 1;
		return 0;
	}
	return 0;
}

void *get_one(uint8_t dtype, uint32_t dl){
	uint8_t one8 = 1;
	uint16_t one16 = 1;
	uint32_t one32 = 1;
	uint64_t one64 = 1;
	float onef = 1.0;
	double oned = 1.0;

	void *one = malloc(dl);
	if (one == NULL) return one;

	if (dtype <= WEKUA_DTYPE_UINT8) memcpy(one, &one8, dl);
	else if (dtype <= WEKUA_DTYPE_UINT16) memcpy(one, &one16, dl);
	else if (dtype <= WEKUA_DTYPE_UINT32) memcpy(one, &one32, dl);
	else if (dtype <= WEKUA_DTYPE_UINT64) memcpy(one, &one64, dl);
	else if (dtype == WEKUA_DTYPE_FLOAT) memcpy(one, &onef, dl);
	else memcpy(one, &oned, dl);

	return one;
}

int wekuaMatrixSum(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be){
	if (a == NULL || (real == NULL && imag == NULL)){
		return CL_INVALID_MEM_OBJECT;
	}
	int ret = CL_SUCCESS;
	wekuaContext ctx = a->ctx;
	uint8_t dtype = a->dtype, com = a->com;
	cl_kernel kernel;
	cl_event e[2], *befo = NULL;
	wmatrix b = NULL, c = NULL;

	uint32_t env = 0;
	uint64_t col, row;

	row = a->shape[0];
	col = a->shape[1];

	if (compileKernel(ctx, WEKUA_KERNEL_SUM, dtype)){
		return CL_COMPILE_PROGRAM_FAILURE;
	}

	kernel = ctx->kernels[WEKUA_KERNEL_SUM*10+dtype];

	clWaitForEvents(nw, be);

	if (row == 1 && col == 1){
		c = a;

		goto wekua_matrix_sum_s2;
	}

	if (com) c = wekuaAllocComplexMatrix(ctx, 1, 1, dtype);
	else c = wekuaAllocMatrix(ctx, 1, 1, dtype);

	if (row == 1){
		b = a;
		goto wekua_matrix_sum_s1;
	}else if (col == 1){
		b = wekuaMatrixTrans(a, 0, NULL, e);
		if (b == NULL) goto wekua_matrix_sum;

		clWaitForEvents(1, e);
		clReleaseEvent(e[0]);

		goto wekua_matrix_sum_s1;
	}

	if (com) b = wekuaAllocComplexMatrix(ctx, 1, a->shape[0], dtype);
	else b = wekuaAllocMatrix(ctx, 1, a->shape[0], dtype);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 8, &a->vl_shape[1]);
	clSetKernelArg(kernel, 5, 1, &a->com);
	
	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &b->shape[1], &b->work_items[7], 0, NULL, e);
	if (ret != CL_SUCCESS) goto wekua_matrix_sum;
	env++;
	befo = e;

	wekua_matrix_sum_s1:
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &c->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &c->imag);
	clSetKernelArg(kernel, 4, 8, &b->vl_shape[1]);
	clSetKernelArg(kernel, 5, 1, &b->com);
	
	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &c->shape[1], &c->work_items[7], env, befo, &e[env]);
	if (ret != CL_SUCCESS){
		if (env > 0){
			clWaitForEvents(env, e);
			clReleaseEvent(e[0]);
		}
		goto wekua_matrix_sum;
	}
	env++;

	wekua_matrix_sum_s2:
	wekuaGetValueFromMatrix(c, 0, 0, real, imag, env, e);

	for (uint32_t x=0; x<env; x++) clReleaseEvent(e[x]);

	wekua_matrix_sum:
	if (a != b) wekuaFreeMatrix(b, 0, NULL);
	if (a != c) wekuaFreeMatrix(c, 0, NULL);
	
	return ret;
}

int wekuaMatrixMul(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be){
	if (a == NULL || (real == NULL && imag == NULL)){
		return CL_INVALID_MEM_OBJECT;
	}
	int ret;
	uint8_t dtype = a->dtype, com = a->com;
	uint64_t col = a->shape[1], row = a->shape[0];
	cl_kernel kernel;
	cl_event e[3];
	wekuaContext ctx = a->ctx;
	wmatrix b=NULL, c=NULL, d=NULL;

	if (compileKernel(ctx, WEKUA_KERNEL_MUL, dtype)){
		return CL_BUILD_PROGRAM_FAILURE;
	}

	kernel = ctx->kernels[WEKUA_KERNEL_MUL*10+dtype];
	
	if (com) b = wekuaAllocComplexMatrix(ctx, row, 1, dtype);
	else b = wekuaAllocMatrix(ctx, row, 1, dtype);

	if (b == NULL){
		ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;
		goto wk_mul_final;
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 8, &col);
	clSetKernelArg(kernel, 5, 8, &a->col);
	clSetKernelArg(kernel, 6, 8, &b->col);
	clSetKernelArg(kernel, 7, 1, &com);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, a->shape, &a->work_items[6], nw, be, e);
	if (ret != CL_SUCCESS) goto wk_mul_final;

	if (row == 1){
		d = b;
		b = NULL;
		clWaitForEvents(1, e);
		clReleaseEvent(e[0]);
		goto wk_mul_jmp;
	}

	c = wekuaMatrixTrans(b, 1, e, &e[1]);

	if (c == NULL){
		ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;
		clWaitForEvents(1, e);
		clReleaseEvent(e[0]);
		goto wk_mul_final;
	}

	if (com) d = wekuaAllocComplexMatrix(ctx, 1, 1, dtype);
	else d = wekuaAllocMatrix(ctx, 1, 1, dtype);

	if (d == NULL){
		ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;
		clWaitForEvents(2, e);
		clReleaseEvent(e[0]);
		clReleaseEvent(e[1]);
		goto wk_mul_final;
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &c->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &c->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &d->imag);
	clSetKernelArg(kernel, 4, 8, &row);
	clSetKernelArg(kernel, 5, 8, &c->col);
	clSetKernelArg(kernel, 6, 8, &d->col);
	clSetKernelArg(kernel, 7, 1, &com);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, c->shape, &c->work_items[6], 1, &e[1], &e[2]);
	if (ret != CL_SUCCESS){
		clWaitForEvents(2, e);
		clReleaseEvent(e[0]);
		clReleaseEvent(e[1]);
		goto wk_mul_final;
	}

	clWaitForEvents(3, e);
	for (uint8_t x=0; x<3; x++) clReleaseEvent(e[x]);

	wk_mul_jmp:
	wekuaGetValueFromMatrix(d, 0, 0, real, imag, 0, NULL);

	wk_mul_final:
	wekuaFreeMatrix(b, 0, NULL);
	wekuaFreeMatrix(c, 0, NULL);
	wekuaFreeMatrix(d, 0, NULL);

	return ret;
}

int wekuaMatrixMean(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be){
	if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	uint8_t dtype = a->dtype;
	uint64_t size = a->shape[0]*a->shape[1];
	int ret = wekuaMatrixSum(a, real, imag, nw, be);

	if (dtype == WEKUA_DTYPE_INT8){
		((int8_t*)real)[0] /= (int8_t)size;
		if (a->com){
			((int8_t*)imag)[0] /= (int8_t)size;
		}
	}else if (dtype == WEKUA_DTYPE_UINT8){
		((uint8_t*)real)[0] /= (uint8_t)size;
		if (a->com){
			((uint8_t*)imag)[0] /= (uint8_t)size;
		}
	}else if (dtype == WEKUA_DTYPE_INT16){
		((int16_t*)real)[0] /= (int16_t)size;
		if (a->com){
			((int16_t*)imag)[0] /= (int16_t)size;
		}
	}else if (dtype == WEKUA_DTYPE_UINT16){
		((int16_t*)real)[0] /= (uint16_t)size;
		if (a->com){
			((int16_t*)imag)[0] /= (uint16_t)size;
		}
	}else if (dtype == WEKUA_DTYPE_INT32){
		((int32_t*)real)[0] /= (int32_t)size;
		if (a->com){
			((int32_t*)imag)[0] /= (int32_t)size;
		}
	}else if (dtype == WEKUA_DTYPE_UINT32){
		((int32_t*)real)[0] /= (uint32_t)size;
		if (a->com){
			((int32_t*)imag)[0] /= (uint32_t)size;
		}
	}else if (dtype == WEKUA_DTYPE_INT64){
		((int64_t*)real)[0] /= (int64_t)size;
		if (a->com){
			((int64_t*)imag)[0] /= (int64_t)size;
		}
	}else if (dtype == WEKUA_DTYPE_UINT64){
		((int64_t*)real)[0] /= size;
		if (a->com){
			((int64_t*)imag)[0] /= size;
		}
	}else if (dtype == WEKUA_DTYPE_FLOAT){
		((float*)real)[0] /= (float)size;
		if (a->com){
			((float*)imag)[0] /= (float)size;
		}
	}else if (dtype == WEKUA_DTYPE_DOUBLE){
		((double*)real)[0] /= (double)size;
		if (a->com){
			((double*)imag)[0] /= (double)size;
		}
	}

	return ret;
}

wmatrix wekuaMatrixPoly(wmatrix a, uint32_t nw, cl_event *be){
	if (a == NULL){
		return NULL;
	}else if (a->shape[1] != a->shape[0]){
		return NULL;
	}

	clWaitForEvents(nw, be);

	int ret;
	uint8_t dtype = a->dtype, com = a->com;
	uint32_t dl, eu = 0;
	wekuaContext ctx = a->ctx;
	uint64_t row;

	wmatrix c = NULL, *b = NULL, i = NULL;
	void *one = NULL, *rs = NULL, *is = NULL;

	if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	dl = ctx->dtype_length[dtype];
	one = get_one(dtype, dl);
	row = a->shape[0];
	rs = calloc(1, dl);
	is = calloc(1, dl);

	if (rs == NULL || is == NULL) goto wk_poly_fail;

	c = wekuaAllocMatrix(a->ctx, 1, row+1, dtype);
	if (c == NULL) return NULL;
	if (com){
		if (createComplexMatrix(c)) goto wk_poly_fail;
	}

	b = (wmatrix*) calloc(2, sizeof(wmatrix));
	if (b == NULL) goto wk_poly_fail;

	b[0] = wekuaAllocMatrix(ctx, row, row, dtype);
	if (b[0] == NULL) goto wk_poly_fail;

	wekuaPutValueToMatrix(c, 0, row, one, NULL, 0, NULL);
	cl_event e[3];

	for (uint64_t x=1; x <= row; x++){
		i = wekuaMatrixIden(ctx, row, dtype);
		if (i == NULL) goto wk_poly_fail;

		b[1] = wekuaAllocMatrix(ctx, row, row, dtype);
		if (b[1] == NULL) goto wk_poly_fail;

		ret = wekuaBlasGemm(one, NULL, 0, a, 0, b[0], NULL, NULL, b[1], 0, NULL);
		if (ret != CL_SUCCESS) goto wk_poly_fail;

		wekuaGetValueFromMatrix(c, 0, row-x+1, rs, is, 0, NULL);
		ret = wekuaBlasScalar(i, rs, is, 0, NULL, &e[1]);
		if (ret == CL_SUCCESS) eu++;
		else goto wk_poly_fail;

		ret = wekuaMatrixAdd(b[1], i, 2, e, &e[2]);
		if (ret == CL_SUCCESS) eu++;
		else goto wk_poly_fail;

		wekuaFreeMatrix(i, 1, &e[2]);
		for (uint8_t l=0; l<3; l++) clReleaseEvent(e[l]);
		eu = 0;

		i = wekuaAllocMatrix(ctx, row, row, dtype);
		if (i == NULL) goto wk_poly_fail;

		ret = wekuaBlasGemm(one, NULL, 0, a, 0, b[1], NULL, NULL, i, 0, NULL);
		if (ret == CL_SUCCESS) goto wk_poly_fail;

		wekuaGetValueFromMatrix(c, 0, row-x, rs, is, 0, NULL);
		ret = wekuaMatrixTrace(i, rs, is, 1, e);
		if (ret == CL_SUCCESS) clReleaseEvent(e[0]);
		else goto wk_poly_fail;

		if (dtype == WEKUA_DTYPE_FLOAT){
			((float*)rs)[0] /= -1.0*x;
			if (com) ((float*)is)[0] /= -1.0*x;
		}else{
			((double*)rs)[0] /= -1.0*x;
			if (com) ((double*)is)[0] /= -1.0*x;
		}
		wekuaPutValueToMatrix(c, 0, row-x, rs, is, 0, NULL);

		wekuaFreeMatrix(i, 0, NULL);
		i = NULL;

		wekuaFreeMatrix(b[0], 0, NULL);

		eu = 0;
		b[0] = b[1];
		b[1] = NULL;
	}

	goto wk_poly_done;

	wk_poly_fail:
	if (c != NULL){
		wekuaFreeMatrix(c, 0, NULL);
		c = NULL;
	}

	wk_poly_done:
	if (b != NULL){
		wekuaFreeMatrix(b[0], 0, NULL);
		wekuaFreeMatrix(b[1], 0, NULL);
		free(b);
	}

	if (eu > 0) clWaitForEvents(eu, e);
	for (uint32_t x=0; x<eu; x++) clReleaseEvent(e[x]);

	if (i != NULL) wekuaFreeMatrix(i, 0, NULL);

	if (rs != NULL) free(rs);
	if (is != NULL) free(is);
	if (one != NULL) free(one);
	return c;	
}

void wekuaMatrixMax(wmatrix a, uint64_t *y, uint64_t *x, uint32_t nw, cl_event *be){
	if (a == NULL || x == NULL || y == NULL) return;

	uint8_t dtype = a->dtype;
	uint64_t i = 0, j = 0;
	uint64_t row = a->shape[0], col = a->shape[1];
	uint32_t dl = a->ctx->dtype_length[dtype];

	void *real, *imag;
	void *r, *im;

	real = malloc(dl); imag = malloc(dl);
	r = malloc(dl); im = malloc(dl);

	wekuaGetValueFromMatrix(a, 0, 0, real, imag, nw, be);

	if (a->com){
		for (uint64_t n=0; n<row; n++){
			for (uint64_t m=0; m<col; m++){
				wekuaGetValueFromMatrix(a, n, m, r, im, 0, NULL);
				if (isgreat(r, real, dtype) && isgreat(im, imag, dtype)){
					memcpy(real, r, dl);
					memcpy(imag, im, dl);
					i = n; j = m;
				}
			}
		}
	}else{
		for (uint64_t n=0; n<row; n++){
			for (uint64_t m=0; m<col; m++){
				wekuaGetValueFromMatrix(a, n, m, r, im, 0, NULL);
				if (isgreat(r, real, dtype)){
					memcpy(real, r, dl);
					i = n; j = m;
				}
			}
		}
	}

	free(real);
	free(imag);
	free(r);
	free(im);

	y[0] = i;
	x[0] = j;
}

void wekuaMatrixMin(wmatrix a, uint64_t *y, uint64_t *x, uint32_t nw, cl_event *be){
	if (a == NULL || x == NULL || y == NULL) return;

	uint8_t dtype = a->dtype;
	uint64_t i = 0, j = 0;
	uint64_t row = a->shape[0], col = a->shape[1];
	uint32_t dl = a->ctx->dtype_length[dtype];

	void *real, *imag;
	void *r, *im;

	real = malloc(dl); imag = malloc(dl);
	r = malloc(dl); im = malloc(dl);

	wekuaGetValueFromMatrix(a, 0, 0, real, imag, nw, be);

	if (a->com){
		for (uint64_t n=0; n<row; n++){
			for (uint64_t m=0; m<col; m++){
				wekuaGetValueFromMatrix(a, n, m, r, im, 0, NULL);
				if (!isgreat(r, real, dtype) && !isgreat(im, imag, dtype)){
					memcpy(real, r, dl);
					memcpy(imag, im, dl);
					i = n; j = m;
				}
			}
		}
	}else{
		for (uint64_t n=0; n<row; n++){
			for (uint64_t m=0; m<col; m++){
				wekuaGetValueFromMatrix(a, n, m, r, im, 0, NULL);
				if (!isgreat(r, real, dtype)){
					memcpy(real, r, dl);
					i = n; j = m;
				}
			}
		}
	}

	free(real);
	free(imag);
	free(r);
	free(im);

	y[0] = i;
	x[0] = j;
}

wmatrix wekuaMatrixEulerIden(wmatrix angle, uint32_t nw, cl_event *be){
	if (angle == NULL) return NULL;
	if (angle->dtype < WEKUA_DTYPE_FLOAT) return NULL;

	uint8_t dtype = angle->dtype;
	wekuaContext ctx = angle->ctx;

	if (compileKernel(ctx, WEKUA_KERNEL_EULER_IDEN, dtype)){
		return NULL;
	}

	int ret;
	uint8_t com = angle->com;
	wmatrix a = wekuaAllocComplexMatrix(ctx, angle->shape[0], angle->shape[1], dtype);
	cl_kernel kernel = ctx->kernels[WEKUA_KERNEL_EULER_IDEN*10 + dtype];
	cl_event e;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &angle->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &angle->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 4, 8, &a->col);
	clSetKernelArg(kernel, 5, 1, &com);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[4], 0, NULL, &e);
	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}else{
		wekuaFreeMatrix(a, 0, NULL);
		a = NULL;
	}

	return a;
}

int wekuaCopyMatrixRegion(
	wmatrix a, uint64_t a_offset_x, uint64_t a_offset_y,
	wmatrix b, uint64_t b_offset_x, uint64_t b_offset_y,
	uint64_t w, uint64_t h
){
	if (a == NULL || b == NULL) return CL_INVALID_MEM_OBJECT;
	if (a_offset_y+h > a->shape[0] || a_offset_x+w > a->shape[1]) return CL_INVALID_ARG_VALUE;
	if (b_offset_y+h > b->shape[0] || b_offset_x+w > b->shape[1]) return CL_INVALID_ARG_VALUE;
	if (a->dtype != b->dtype) return CL_INVALID_MEM_OBJECT;

	cl_event e;
	wekuaContext ctx = a->ctx;
	uint32_t dl = ctx->dtype_length[a->dtype];
	uint8_t com = a->com;
	int ret;

	uint64_t region[3];
	uint64_t src_origin[3], dst_origin[3];

	src_origin[0] = a_offset_x;
	src_origin[1] = a_offset_y;
	src_origin[2] = 0;

	dst_origin[0] = b_offset_x;
	dst_origin[1] = b_offset_y;
	dst_origin[2] = 0;

	region[0] = w*dl;
	region[1] = h;
	region[2] = 1;

	if (com){
		if (createComplexMatrix(b)){
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
	}

	if (ctx == b->ctx){
		ret = clEnqueueCopyBufferRect(
			ctx->command_queue,
			a->real, b->real,
			src_origin, dst_origin,
			region, a->col*dl, 0,
			b->col*dl, 0, 0, NULL, &e
		);
		if (ret != CL_SUCCESS) return ret;

		clWaitForEvents(1, &e);
		clReleaseEvent(e);
		
		if (com){
			ret = clEnqueueCopyBufferRect(
				ctx->command_queue,
				a->imag, b->imag,
				src_origin, dst_origin,
				region, a->col*dl, 0,
				b->col*dl, 0, 0, NULL, &e
			);
			if (ret != CL_SUCCESS) return ret;

			clWaitForEvents(1, &e);
			clReleaseEvent(e);
		}
	}else{
		ret = clEnqueueWriteBufferRect(
			ctx->command_queue,
			a->real, CL_TRUE,
			src_origin, dst_origin,
			region, a->col*dl, 0,
			b->col*dl, 0, b->raw_real, 0, NULL, NULL
		);
		if (ret != CL_SUCCESS) return ret;
		if (com){
			ret = clEnqueueWriteBufferRect(
				ctx->command_queue,
				a->imag, CL_TRUE,
				src_origin, dst_origin,
				region, a->col*dl, 0,
				b->col*dl, 0, b->raw_imag, 0, NULL, NULL
			);
			if (ret != CL_SUCCESS) return ret;
		}
	}

	return ret;
}
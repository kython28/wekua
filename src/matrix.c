#include "../headers/matrix.h"
#include "buffer.h"
#include <math.h>

uint64_t zero = 0;

void getLWI(uint64_t *x, uint64_t *y, uint32_t si, uint64_t max){
	uint64_t c = (uint64_t)(pow(1.0*max, 1.0/si));
	uint64_t a, b;
	for (uint32_t j=0; j<si; j++){
		a = x[j];
		if (a < c){
			y[j] = a;
			continue;
		}
		b = c;
		while (a%b != 0){
			b--;
		}
		x[j] = a; y[j] = b;
	}
}

void *get_one(uint8_t dtype, uint32_t dl);

int MapBufferMatrix(wmatrix a){
	if (a == NULL) return CL_INVALID_MEM_OBJECT;

	int ret = CL_SUCCESS;
	cl_command_queue cmd = a->ctx->command_queue;
	cl_event e[2];
	uint32_t nw = 0;
	uint64_t size = a->length;

	if (a->real != NULL && a->raw_real == NULL){
		a->raw_real = clEnqueueMapBuffer(cmd, a->real, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, size, 0, 0, e, &ret);
		if (ret == CL_SUCCESS) nw++;
	}
	if (a->imag != NULL && a->raw_imag == NULL && ret == CL_SUCCESS){
		a->raw_imag = clEnqueueMapBuffer(cmd, a->imag, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, size, 0, 0, &e[nw], &ret);
		if (ret == CL_SUCCESS) nw++;
	}

	if (nw > 0){
		clWaitForEvents(nw, e);
		for (uint32_t x=0; x<nw; x++) clReleaseEvent(e[x]);
	}

	return ret;
}

int UnmapBufferMatrix(wmatrix a){
	if (a == NULL) return CL_INVALID_MEM_OBJECT;

	int ret = CL_SUCCESS;
	cl_event e;

	cl_command_queue cmd = a->ctx->command_queue;
	if (a->real != NULL && a->raw_real != NULL){
		ret = clEnqueueUnmapMemObject(cmd, a->real, a->raw_real, 0, NULL, &e);
		if (ret == CL_SUCCESS){
			clWaitForEvents(1, &e);
			clReleaseEvent(e);
			if (ret == CL_SUCCESS) a->raw_real = NULL;
		}
	}
	if (a->imag != NULL && a->raw_imag != NULL && ret == CL_SUCCESS){
		ret = clEnqueueUnmapMemObject(cmd, a->imag, a->raw_imag, 0, NULL, &e);
		if (ret == CL_SUCCESS){
			clWaitForEvents(1, &e);
			clReleaseEvent(e);
			if (ret == CL_SUCCESS) a->raw_imag = NULL;
		}
	}
	return ret;
}

static int free_matrix(void *ptr){
	wmatrix a = ptr;
	int ret = UnmapBufferMatrix(a);
	if (a->real != NULL && ret == CL_SUCCESS){
		ret = clReleaseMemObject(a->real);
		if (ret == CL_SUCCESS) a->real = NULL;
	}
	if (a->imag != NULL && ret == CL_SUCCESS){
		ret = clReleaseMemObject(a->imag);
	}
	if (ret == CL_SUCCESS) free(a);
	return ret;
}

int mem_set_zero(wmatrix a, cl_mem buf){
	int ret = CL_SUCCESS;
	cl_event e;

	wekuaContext ctx = a->ctx;

	ret = clEnqueueFillBuffer(ctx->command_queue, buf, &zero, ctx->dtype_length[a->dtype], 0, a->length, 0, NULL, &e);
	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}
	
	return ret;
}

uint8_t createComplexMatrix(wmatrix b){
	if (b == NULL){
		return 1;
	}else if (b->com){
		return 0;
	}
	wekuaContext ctx = b->ctx;
	uint64_t size = b->length;
	int ret;

	b->imag = wekuaCreateBuffer(ctx, size, &ret);
	if (ret != CL_SUCCESS){
		return 1;
	}
	MapBufferMatrix(b);
	mem_set_zero(b, b->imag);
	b->com = 1;
	return 0;
}

int removeComplexMatrix(wmatrix b, uint32_t nw, cl_event *be){
	if (b == NULL){
		return CL_INVALID_MEM_OBJECT;
	}else if (b->com == 0){
		return CL_SUCCESS;
	}

	cl_event e;
	int ret;
	
	clWaitForEvents(nw, be);
	
	ret = clEnqueueUnmapMemObject(b->ctx->command_queue, b->imag, b->raw_imag, 0, NULL, &e);
	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
		b->raw_imag = NULL;

		ret  = clReleaseMemObject(b->imag);
		if (ret == CL_SUCCESS){
			b->imag = NULL;
			b->com = 0;
		}
	}
	return ret;
}

void wekuaFreeMatrix(wmatrix a, uint32_t nw, cl_event *be){
	if (a == NULL) return;

	clWaitForEvents(nw, be);
	wekuaFIFOPut(a->ctx->garbage_queue, a);
	
}

wmatrix wekuaMatrixEmpty(wekuaContext ctx, uint64_t r, uint64_t c, uint8_t dtype){
	if (ctx == NULL || r == 0 || c == 0 || dtype > 9){
		return NULL;
	}
	wmatrix a = (wmatrix) calloc(1, sizeof(struct _wk_matrix));
	if (a == NULL){
		return NULL;
	}

	uint64_t max = ctx->max_work_group_size;
	uint32_t vl = ctx->vector_width[dtype], dl = ctx->dtype_length[dtype];

	a->ctx = ctx;
	a->dtype = dtype;

	a->free = &free_matrix;

	
	a->shape[0] = r;
	a->shape[1] = c;

	if (c%vl != 0) c += vl - c%vl;
	if ((c/vl)%2 != 0) c += vl;
	if (r%2 != 0) r++;

	a->col = c;
	a->row = r;

	a->size = r*c;
	a->length = r*c*dl;

	c /= vl;

	a->vl_shape[0] = r;
	a->vl_shape[1] = c;

	a->vl_shape[2] = r*c;

	getLWI(a->vl_shape, a->work_items, 2, max);
	getLWI(a->vl_shape, &a->work_items[2], 1, max);
	getLWI(&a->vl_shape[1], &a->work_items[3], 1, max);

	getLWI(a->shape, &a->work_items[4], 2, max);
	getLWI(a->shape, &a->work_items[6], 1, max);
	getLWI(&a->shape[1], &a->work_items[7], 1, max);

	getLWI(&a->vl_shape[2], &a->work_items[8], 1, max);

	int ret;

	a->real = wekuaCreateBuffer(ctx, a->length, &ret);
	if (ret != CL_SUCCESS){
		free(a);
		return NULL;
	}
	ret = MapBufferMatrix(a);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(a, 0, NULL);
		a = NULL;
	}
	return a;
}

wmatrix wekuaAllocMatrix(wekuaContext ctx, uint64_t r, uint64_t c, uint8_t dtype){
	wmatrix a = wekuaMatrixEmpty(ctx, r, c, dtype);
	if (a == NULL){
		return NULL;
	}

	if (mem_set_zero(a, a->real) != CL_SUCCESS){
		wekuaFreeMatrix(a, 0, NULL);
		a = NULL;
	}
	return a;
}

wmatrix wekuaAllocComplexMatrix(wekuaContext ctx, uint64_t r, uint64_t c, uint8_t dtype){
	wmatrix a = wekuaAllocMatrix(ctx, r, c, dtype);
	if (createComplexMatrix(a)){
		wekuaFreeMatrix(a, 0, NULL);
		return NULL;
	}
	return a;
}

wmatrix wekuaFillMatrix(wekuaContext ctx, uint64_t r, uint64_t c, void *alpha, void *beta, uint8_t dtype){
	wmatrix a = wekuaAllocMatrix(ctx, r, c, dtype);
	if (a == NULL) return NULL;
	
	int ret;
	uint8_t com = 0;
	uint32_t dl = ctx->dtype_length[dtype], evn = 0;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_FILL, dtype, com);
	if (kernel == NULL){
		wekuaFreeMatrix(a, 0, NULL);
		return NULL;
	}

	cl_event e;

	if (beta == NULL) beta = &zero;
	if (alpha == NULL) alpha = &zero;

	if (memcmp(&zero, beta, dl) != 0){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, dl, alpha);
	clSetKernelArg(kernel, 3, dl, beta);
	clSetKernelArg(kernel, 4, 8, &a->col);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[4], 0, NULL, &e);
	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}

	return a;
}

wmatrix wekuaMatrixRandn(wekuaContext ctx, uint64_t r, uint64_t c, uint8_t com){
	wmatrix a = wekuaAllocMatrix(ctx, r, c, WEKUA_DTYPE_DOUBLE);
	if (a == NULL) return NULL;

	if (com){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
	}
	uint8_t dtype = WEKUA_DTYPE_DOUBLE;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_RANDN, dtype, com);
	if (kernel == NULL){
		wekuaFreeMatrix(a, 0, NULL);
		return NULL;
	}

	cl_command_queue cmd = ctx->command_queue;
	cl_context ct = ctx->ctx;
	cl_mem ran_r=NULL, ran_i=NULL;
	cl_event e;

	uint64_t *ran_r_m, *ran_i_m, size = a->size*8;

	int ret;

	ran_r = wekuaCreateBuffer(ctx, size, &ret);
	if (ret != CL_SUCCESS){
		clReleaseMemObject(ran_r);
		return NULL;
	}
	ran_r_m = clEnqueueMapBuffer(cmd, ran_r, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, size, 0, 0, &e, NULL);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	getRandomBuffer(ran_r_m, size);
	clEnqueueUnmapMemObject(cmd, ran_r, ran_r_m, 0, NULL, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	if (com){
		ran_i = wekuaCreateBuffer(ctx, size, &ret);
		if (ret != 0){
			clReleaseMemObject(ran_r);
			clReleaseMemObject(ran_i);
			return NULL;
		}
		ran_i_m = clEnqueueMapBuffer(cmd, ran_i, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, size, 0, 0, &e, NULL);
		clWaitForEvents(1, &e);
		clReleaseEvent(e);

		getRandomBuffer(ran_i_m, size);
		clEnqueueUnmapMemObject(cmd, ran_i, ran_i_m, 0, NULL, &e);
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &ran_r);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &ran_i);
	clSetKernelArg(kernel, 4, 8, &a->col);
	
	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[4], 0, NULL, &e);
	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}else{
		wekuaFreeMatrix(a, 0, NULL);
		a = NULL;
	}

	clReleaseMemObject(ran_r);
	clReleaseMemObject(ran_i);

	return a;
}

wmatrix wekuaMatrixRandUniform(wekuaContext ctx, uint64_t r, uint64_t c, void *ra, void *ia, void *re, void *ie, uint8_t dtype){
	if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	uint8_t com = 0;
	uint32_t dl = ctx->dtype_length[dtype], evn = 0;
	wmatrix a, b;
	cl_event e[2], *befo = NULL;

	if (ia == NULL) ia = &zero;
	if (ie == NULL) ie = &zero;
	if (ra == NULL) ra = &zero;
	if (re == NULL) re = &zero;

	if (memcmp(ia, &zero, dl) != 0 || memcmp(ie, &zero, dl) != 0) com = 1;

	a = wekuaMatrixRandn(ctx, r, c, com);
	if (dtype == WEKUA_DTYPE_FLOAT){
		b = wekuaMatrixConvert(a, WEKUA_DTYPE_FLOAT, 0, NULL, e);
		befo = e;
		evn++;
	}else{
		b = a;
	}

	if (b == NULL){
		wekuaFreeMatrix(a, 0, NULL);
		return NULL;
	}

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_RANDUNIFORM, dtype, com);
	if (kernel == NULL){
		if (a != b) wekuaFreeMatrix(b, evn, befo);
		wekuaFreeMatrix(a, evn, befo);
		return NULL;
	}

	if (com){
		clWaitForEvents(1, e);
		if (createComplexMatrix(b)){
			if (a != b) wekuaFreeMatrix(b, 0, NULL);
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, dl, ra);
	clSetKernelArg(kernel, 3, dl, ia);
	clSetKernelArg(kernel, 4, dl, re);
	clSetKernelArg(kernel, 5, dl, ie);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &b->vl_shape[2], &b->work_items[8], evn, befo, &e[evn]);
	if (ret == CL_SUCCESS){
		evn++;
		clWaitForEvents(evn, e);
		if (befo != NULL) clReleaseEvent(befo[0]);
		clReleaseEvent(e[evn-1]);
	}else{
		wekuaFreeMatrix(a, evn, befo);
		wekuaFreeMatrix(b, 0, NULL);
		a = NULL;
		b = NULL;

		clReleaseEvent(befo[0]);
	}

	if (a != b){
		wekuaFreeMatrix(a, 0, NULL);
	}

	return b;
}

wmatrix wekuaMatrixCopy(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return NULL;
	}

	wmatrix b = (wmatrix) calloc(1, sizeof(struct _wk_matrix));
	if (b == NULL) return NULL;

	wekuaContext ctx = a->ctx;
	uint64_t length = a->length;
	uint8_t com = a->com;

	b->dtype = a->dtype;
	b->length = length;
	b->ctx = ctx;
	b->com = com;
	b->row = a->row;
	b->col = a->col;
	b->size = a->size;
	b->free = &free_matrix;

	memcpy(b->shape, a->shape, 16);
	memcpy(b->vl_shape, a->vl_shape, 24);
	memcpy(b->work_items, a->work_items, 72);

	int ret;
	b->real = wekuaCreateBuffer(ctx, length, &ret);
	if (ret != CL_SUCCESS){
		free(b);
		return NULL;
	}
	ret = clEnqueueCopyBuffer(ctx->command_queue, a->real, b->real, 0, 0, length, nw, be, e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(b, 0, NULL);
		return NULL;
	}
	if (com && b){
		b->imag = wekuaCreateBuffer(ctx, length, &ret);
		if (ret != CL_SUCCESS){
			wekuaFreeMatrix(b, 0, NULL);
			return NULL;
		}

		clWaitForEvents(1, e);
		clReleaseEvent(e[0]);
		ret = clEnqueueCopyBuffer(ctx->command_queue, a->imag, b->imag, 0, 0, length, 0, NULL, e);
		if (ret != CL_SUCCESS){
			wekuaFreeMatrix(b, 0, NULL);
			b = NULL;
		}
	}

	if (MapBufferMatrix(b) != CL_SUCCESS){
		wekuaFreeMatrix(b, 0, NULL);
		b = NULL;
	}
	return b;
}

wmatrix wekuaMatrixResize(wmatrix a, uint64_t r, uint64_t c, void *alpha, void *beta){
	if (a == NULL) return NULL;
	wekuaContext ctx = a->ctx;
	uint8_t dtype = a->dtype, com;
	cl_command_queue cmd = ctx->command_queue;


	wmatrix b = wekuaFillMatrix(ctx, r, c, alpha, beta, dtype);
	if (b == NULL) return NULL;
	com = b->com;

	if (com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b, 0, NULL);
			return NULL;
		}
	}

	uint64_t col = a->col, col2 = b->col;
	uint64_t width = ctx->dtype_length[dtype], posi, posi2, si, we = 0, h;
	si = a->shape[1];
	if (si > c){
		si = c;
	}
	si *= width;

	h = a->shape[0];
	if (h > r){
		h = r;
	}


	int ret;

	cl_event *e = calloc(h, sizeof(cl_event));
	for (uint64_t i=0; i<h; i++){
		posi = i*col*width;
		posi2 = i*col2*width;
		ret = clEnqueueCopyBuffer(cmd, a->real, b->real, posi, posi2, si, 0, NULL, &e[i]);
		if (ret == CL_SUCCESS) we++;
		else {
			if (we > 0) clWaitForEvents(1, &e[we-1]);
			wekuaFreeMatrix(b, 0, NULL);
			b = NULL;
			break;
		}
		if (com){
			clWaitForEvents(1, &e[i]);
			clReleaseEvent(e[i]);

			ret = clEnqueueCopyBuffer(cmd, a->imag, b->imag, posi, posi2, si, 0, NULL, &e[i]);
			if (ret != CL_SUCCESS){
				if (we > 0) we--;
				if (i > 0) clWaitForEvents(1, &e[i-1]);
				wekuaFreeMatrix(b, 0, NULL);
				b = NULL;
				break;
			}
		}
	}

	if (we > 0) clWaitForEvents(1, &e[we-1]);
	for (uint64_t i=0; i<we; i++){
		clReleaseEvent(e[i]);
	}
	free(e);
	return b;
}

wmatrix wekuaMatrixConvert(wmatrix a, uint8_t dtype, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL) return NULL;
	if (a->dtype == dtype) return wekuaMatrixCopy(a, nw, be, e);
	uint8_t com = a->com;
	uint64_t col = a->shape[1], row = a->shape[0];
	wekuaContext ctx = a->ctx;
	cl_kernel kernel;
	wmatrix b;

	kernel = compileKernel(ctx, WEKUA_KERNEL_CONVERT, dtype, com);
	if (kernel == NULL) return NULL;

	if (com){
		b = wekuaAllocComplexMatrix(ctx, row, col, dtype);
	}else{
		b = wekuaAllocMatrix(ctx, row, col, dtype);
	}

	if (b == NULL) return NULL;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &a->imag);

	clSetKernelArg(kernel, 4, 8, &b->col);
	clSetKernelArg(kernel, 5, 8, &a->col);
	clSetKernelArg(kernel, 6, 1, &a->dtype);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, b->shape, &b->work_items[4], nw, be, e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(b, 0, NULL);
		b = NULL;
	}

	return b;
}

wmatrix wekuaMatrixFromBuffer(wekuaContext ctx, uint64_t r, uint64_t c, void *rbuf, void *ibuf, uint8_t dtype){
	if (ctx == NULL || r == 0 || c == 0) return NULL;
	else if (rbuf == NULL && ibuf == NULL) return NULL;

	wmatrix a = wekuaAllocMatrix(ctx, r, c, dtype);
	uint64_t length = a->length, buff_ori[3] = {0}, region[3];
	uint32_t dl = ctx->dtype_length[dtype];

	cl_event e[2];
	uint32_t events_num = 1;

	region[0] = c*dl;
	region[1] = r;
	region[2] = 1;

	clEnqueueWriteBufferRect(
		ctx->command_queue, a->real, CL_FALSE, buff_ori,
		buff_ori, region, a->col*dl, a->length, c*dl,
		0, rbuf, 0, NULL, e
	);


	if (ibuf != NULL){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
		clEnqueueWriteBufferRect(
			ctx->command_queue, a->imag, CL_FALSE, buff_ori,
			buff_ori, region, a->col*dl, a->length, c*dl,
			0, ibuf, 0, NULL, &e[1]
		);

		events_num++;
	}

	clWaitForEvents(events_num, e);
	for (uint32_t x=0; x<events_num; x++) clReleaseEvent(e[x]);

	return a;
}

wmatrix wekuaMatrixIden(wekuaContext ctx, uint64_t col, uint8_t dtype){
	if (ctx == NULL || col == 0 || dtype > 9) return NULL;
	wmatrix iden = wekuaAllocMatrix(ctx, col, col, dtype);
	if (iden == NULL) return iden;
	int ret;

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_IDEN, dtype, 0);
	if (kernel == NULL) return NULL;
	cl_event e;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &iden->real);
	clSetKernelArg(kernel, 1, 8, &iden->col);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, iden->shape, &iden->work_items[6], 0, NULL, &e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(iden, 0, NULL);
		iden = NULL;
	}else{
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}

	return iden;
}

wmatrix wekuaMatrixTrans(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL) return NULL;

	wekuaContext ctx = a->ctx;
	uint64_t *shape = a->shape;
	uint8_t dtype = a->dtype, com = a->com;
	int ret;
	
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_TRANS, dtype, com);
	if (kernel == NULL) return NULL;

	wmatrix b = wekuaAllocMatrix(ctx, shape[1], shape[0], dtype);
	if (com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b, 0, NULL);
			return NULL;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
 
	clSetKernelArg(kernel, 4, 8, &a->col);
	clSetKernelArg(kernel, 5, 8, &b->col);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, shape, &a->work_items[4], nw, be, e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(b, 0, NULL);
		b = NULL;
	}
	return b;
}

wmatrix wekuaMatrixDiag(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return NULL;
	}

	wekuaContext ctx = a->ctx;
	uint8_t mode = 0, com, dtype = a->dtype;
	uint64_t *shapea = a->shape, col;
	int ret;
	wmatrix b;

	cl_kernel kernel;

	col = shapea[1];
	com = a->com;

	if (shapea[0] == shapea[1]){
		if (com){
			b = wekuaAllocComplexMatrix(ctx, 1, col, dtype);
		}else{
			b = wekuaAllocMatrix(ctx, 1, col, dtype);
		}
		mode = 1;
	}else if (shapea[0] == 1){
		if (com){
			b = wekuaAllocComplexMatrix(ctx, col, col, dtype);
		}else{
			b = wekuaAllocMatrix(ctx, col, col, dtype);
		}
	}else{
		return NULL;
	}

	if (b == NULL) return NULL;

	kernel = compileKernel(ctx, WEKUA_KERNEL_DIAG, dtype, com);
	if (kernel == NULL){
		wekuaFreeMatrix(b, 0, NULL);
		return NULL;
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 8, &a->col);
	clSetKernelArg(kernel, 5, 1, &mode);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &col, &a->work_items[7], nw, be, e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(b, 0, NULL);
		b = NULL;
	}

	return b;
}

wmatrix wekuaMatrixAbs(wmatrix a, uint32_t nw, cl_event *be){
	if (a == NULL) return NULL;

	cl_event e[2];
	wekuaContext ctx = a->ctx;
	wmatrix b;
	uint8_t com = a->com, dtype;
	cl_kernel kernel;

	if (com){
		b = wekuaMatrixConvert(a, WEKUA_DTYPE_DOUBLE, nw, be, e);
		dtype = WEKUA_DTYPE_DOUBLE;
	}else{
		b = wekuaMatrixCopy(a, nw, be, e);
		dtype = a->dtype;
	}
	if (b == NULL) return NULL;

	kernel = compileKernel(ctx, WEKUA_KERNEL_ABS, dtype, com);
	if (kernel == NULL){
		wekuaFreeMatrix(b, 1, e);
		clReleaseEvent(e[0]);
		return NULL;
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, 8, &a->vl_shape[1]);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, b->vl_shape, a->work_items, 1, e, &e[1]);
	if (ret == CL_SUCCESS){
		removeComplexMatrix(b, 1, &e[1]);

		clReleaseEvent(e[0]);
		clReleaseEvent(e[1]);
	}else{
		wekuaFreeMatrix(b, 1, e);
		clReleaseEvent(e[0]);
		b = NULL;
	}

	return b;
}

wmatrix wekuaMatrixAbsdiff(wmatrix a, wmatrix b, uint32_t nw, cl_event *be){
	cl_event e[2];
	wmatrix c, d;
	int ret;

	c = wekuaMatrixCopy(a, nw, be, e);
	if (c == NULL) return NULL;

	ret = wekuaMatrixSub(c, b, 1, e, &e[1]);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(c, 1, e);
		clReleaseEvent(e[0]);
		return NULL;
	}

	d = wekuaMatrixAbs(c, 1, &e[1]);
	if (d == NULL) clWaitForEvents(1, &e[1]);

	clReleaseEvent(e[0]);
	clReleaseEvent(e[1]);
	wekuaFreeMatrix(c, 0, NULL);
	return d;
}

wmatrix wekuaMatrixArange(wekuaContext ctx,
	double start_r, double start_i, double end_r, double end_i,
	double delta, uint8_t trans
){
	if (start_r == end_r && start_i == end_i) return NULL;

	wmatrix a;
	uint8_t com;
	uint64_t col;
	double xdis, ydis, norm, theta;
	double delta_r, delta_i;
	int ret;

	cl_event e;
	cl_kernel kernel;

	xdis = end_r-start_r;
	ydis = end_i-start_i;

	norm = sqrt(xdis*xdis + ydis*ydis);
	if (xdis == 0.0) theta = CL_M_PI_2;
	else theta = atan(ydis/xdis);

	col = (uint64_t) norm/delta;
	while (col*delta > norm) col--;

	delta_r = (xdis*delta)/(cos(theta)*norm);
	delta_i = (ydis*delta)/(sin(theta)*norm);

	if (ydis == 0.0) com = 0;
	else com = 1;

	if (trans){
		a = wekuaAllocMatrix(ctx, col, 1, WEKUA_DTYPE_DOUBLE);
	}else{
		a = wekuaAllocMatrix(ctx, 1, col, WEKUA_DTYPE_DOUBLE);
	}

	if (a == NULL) return NULL;

	if (com){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
	}

	kernel = compileKernel(ctx, WEKUA_KERNEL_ARANGE, WEKUA_DTYPE_DOUBLE, com);
	if (kernel == NULL){
		wekuaFreeMatrix(a, 0, NULL);
		return NULL;
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);

	clSetKernelArg(kernel, 2, sizeof(double), &start_r);
	clSetKernelArg(kernel, 3, sizeof(double), &start_i);
	clSetKernelArg(kernel, 4, sizeof(double), &delta_r);
	clSetKernelArg(kernel, 5, sizeof(double), &delta_i);
	clSetKernelArg(kernel, 6, sizeof(double), &delta);
	clSetKernelArg(kernel, 7, sizeof(double), &theta);

	clSetKernelArg(kernel, 8, 8, &a->col);
	clSetKernelArg(kernel, 9, 1, &trans);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[4], 0, NULL, &e);
	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(a, 0, NULL);
		a = NULL;
	}else{
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}

	return a;
}

wmatrix wekuaMatrixInv(wmatrix a, uint32_t nw, cl_event *be){
	if (a == NULL) return NULL;
	else if (a->shape[0] != a->shape[1]) return NULL;
	else if (a->dtype < WEKUA_DTYPE_FLOAT) return NULL;

	wekuaContext ctx = a->ctx;
	uint64_t col = a->shape[1], rcol = a->vl_shape[1], evn, *shape = a->vl_shape, *wi = a->work_items;
	uint8_t dtype = a->dtype, otherm = 1, com = a->com;
	uint32_t dl = ctx->dtype_length[dtype];
	int ret;
	wmatrix inv = NULL, b = NULL;
	cl_event *e = NULL;
	cl_kernel kernel;

	cl_mem *b_real, *b_imag, *i_real, *i_imag;

	if (compileKernel(ctx, WEKUA_KERNEL_GAUSS, dtype, com) == NULL) return NULL;
	if (compileKernel(ctx, WEKUA_KERNEL_GAUSS_2, dtype, com) == NULL) return NULL;

	kernel = compileKernel(ctx, WEKUA_KERNEL_GAUSS, dtype, com);

	e = (cl_event*) calloc(col+2, sizeof(cl_event));
	if (e == NULL) goto wekua_inv_fail;

	inv = wekuaMatrixIden(ctx, col, dtype);
	if (inv == NULL) goto wekua_inv_fail;

	b = wekuaMatrixCopy(a, nw, be, e);
	if (b == NULL) goto wekua_inv_fail;

	if (com){
		if (createComplexMatrix(inv)) goto wekua_inv_fail;
	}

	b_real = &b->real;
	b_imag = &b->imag;

	i_real = &inv->real;
	i_imag = &inv->imag;

	for (evn = 0; evn < col;){
		clSetKernelArg(kernel, 0, sizeof(cl_mem), b_real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), b_imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), i_real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), i_imag);
		clSetKernelArg(kernel, 4, 8, &evn);
		clSetKernelArg(kernel, 5, 8, &rcol);
		clSetKernelArg(kernel, 6, 1, &otherm);
		clSetKernelArg(kernel, 7, 1, &otherm);

		ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, shape, &wi[2], 1, &e[evn], &e[evn+1]);
		if (ret != CL_SUCCESS) goto wekua_inv_fail;
		evn++;
	}

	kernel = compileKernel(ctx, WEKUA_KERNEL_GAUSS_2, dtype, com);

	dl *= wi[4];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), i_real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), i_imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), b_real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), b_imag);
	clSetKernelArg(kernel, 4, 8, &rcol);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, shape, wi, evn, e, &e[evn+1]);
	if (ret != CL_SUCCESS) goto wekua_inv_fail;

	evn++;

	wekua_inv_fail:
	if (e != NULL){
		evn++;
		clWaitForEvents(evn, e);
		for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);
		free(e);
	}
	wekuaFreeMatrix(b, 0, NULL);

	return inv;
}

wmatrix wekuaMatrixSolve(wmatrix a, wmatrix b, uint32_t nw, cl_event *be){
	if (a == NULL || b == NULL) return NULL;
	else if (a->dtype != b->dtype) return NULL;
	else if (a->dtype < WEKUA_DTYPE_FLOAT) return NULL;

	wekuaContext ctx = a->ctx;
	uint8_t dtype = a->dtype;

	void *one = get_one(dtype, ctx->dtype_length[dtype]);
	if (one == NULL) return NULL;

	wmatrix x, c;

	c = wekuaMatrixInv(a, nw, be);
	if (c == NULL) return NULL;

	x = wekuaAllocMatrix(ctx, a->shape[0], b->shape[1], dtype);
	if (x == NULL){
		free(one);
		wekuaFreeMatrix(c, 0, NULL);
		return NULL;
	}

	if (wekuaBlasGemm(one, NULL, 0, c, 0, b, NULL, NULL, x, 0, NULL) != CL_SUCCESS){
		free(one);
		wekuaFreeMatrix(c, 0, NULL);
		wekuaFreeMatrix(x, 0, NULL);
		return NULL;
	}

	free(one);
	wekuaFreeMatrix(c, 0, NULL);

	return x;
}

wmatrix wekuaMatrixPinv(wmatrix a, uint32_t nw, cl_event *be){
	if (a == NULL) return NULL;
	else if (a->dtype < WEKUA_DTYPE_FLOAT) return NULL;

	wekuaContext ctx = a->ctx;
	uint8_t dtype = a->dtype;
	void *one = get_one(dtype, ctx->dtype_length[dtype]);
	cl_event e;
	wmatrix pinv=NULL, te=NULL, ti=NULL;

	uint32_t rang = wekuaMatrixRang(a, 1, &e);

	clReleaseEvent(e);

	te = wekuaAllocMatrix(ctx, rang, rang, dtype);

	if (rang == a->shape[0]){
		wekuaBlasGemm(one, NULL, 0, a, 1, a, NULL, NULL, te, 0, NULL);
		
		ti = wekuaMatrixInv(te, 0, NULL);

		wekuaBlasGemm(one, NULL, 1, a, 0, ti, NULL, NULL, pinv, 0, NULL);
	}else if (rang == a->shape[1]){
		wekuaBlasGemm(one, NULL, 1, a, 0, a, NULL, NULL, te, 0, NULL);

		ti = wekuaMatrixInv(te, 1, &e);
		wekuaBlasGemm(one, NULL, 0, ti, 1, a, NULL, NULL, pinv, 0, NULL);
	}
	wekuaFreeMatrix(te, 0, NULL);
	wekuaFreeMatrix(ti, 0, NULL);

	free(one);

	return pinv;
}

uint64_t wekuaMatrixRang(wmatrix a, uint32_t nw, cl_event *be){
	if (a == NULL) return 0;
	else if (a->dtype < WEKUA_DTYPE_FLOAT) return 0;

	wekuaContext ctx = a->ctx;
	int ret;
	uint8_t dtype = a->dtype, otherm = 0, com = a->com;
	uint64_t rang = 0, evn, r = a->shape[0], *wi = &a->work_items[6], col, rcol, *shape = a->shape;
	uint32_t dl = ctx->dtype_length[dtype];
	wmatrix b = NULL, c = NULL, d = NULL, f = NULL;
	cl_event *e = NULL;
	cl_kernel kernel;
	void *one = get_one(dtype, dl);

	kernel = compileKernel(ctx, WEKUA_KERNEL_GAUSS, dtype, com);
	if (kernel == NULL) goto wekua_rang_fail;

	if (r > a->shape[1]) r = a->shape[1];
	col = a->shape[1];
	rcol = a->col;

	e = (cl_event*) calloc(r, sizeof(cl_event));
	if (e == NULL) goto wekua_rang_fail;

	b = wekuaMatrixCopy(a, nw, be, e);
	if (b == NULL) goto wekua_rang_fail;

	c = wekuaFillMatrix(ctx, 1, b->shape[1], one, NULL, dtype);
	if (c == NULL) goto wekua_rang_fail;

	d = wekuaAllocMatrix(ctx, b->shape[0], 1, dtype);
	if (d == NULL) goto wekua_rang_fail;

	for (evn = 0; evn < (r-1); evn++){
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), NULL);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), NULL);
		clSetKernelArg(kernel, 4, 8, &evn);
		clSetKernelArg(kernel, 5, 8, &col);
		clSetKernelArg(kernel, 6, 8, &rcol);
		clSetKernelArg(kernel, 7, 1, &otherm);
		clSetKernelArg(kernel, 8, 1, &otherm);

		ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, shape, wi, 1, &e[evn], &e[evn + 1]);
		if (ret != CL_SUCCESS) goto wekua_rang_fail;
	}

	f = wekuaMatrixAbs(b, 1, &e[evn]);
	if (f == NULL) goto wekua_rang_fail;

	wekuaBlasGemm(one, NULL, 0, f, 1, c, NULL, NULL, d, 0, NULL);

	for (uint64_t i=0; i<d->shape[0]; i++){
		wekuaGetValueFromMatrix(d, i, 0, one, NULL, 0, NULL);
		if (memcmp(one, &zero, dl)) rang++;
	}

	wekua_rang_fail:
	if (e != NULL){
		evn++;
		clWaitForEvents(evn, e);
		free(e);
	}
	if (one != NULL) free(one);
	wekuaFreeMatrix(b, 0, NULL);
	wekuaFreeMatrix(c, 0, NULL);
	wekuaFreeMatrix(d, 0, NULL);
	wekuaFreeMatrix(f, 0, NULL);

	return rang;
}

int wekuaMatrixAdd(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL || b == NULL){
		return CL_INVALID_MEM_OBJECT;
	}else if (a->dtype != b->dtype){
		return CL_INVALID_MEM_OBJECT;
	}

	void *alpha;
	uint8_t cone = 1, dtype = a->dtype;
	uint16_t sone = 1;
	uint32_t ione = 1;
	uint64_t lone = 1;
	float fone = 1.0;
	double done = 1.0;

	if (dtype == 9) alpha = &done;
	else if (dtype == 8) alpha = &fone;
	else if (dtype >= 6) alpha = &lone;
	else if (dtype >= 4) alpha = &ione;
	else if (dtype >= 2) alpha = &sone;
	else alpha = &cone;

	return wekuaBlasAxpy(b, a, alpha, NULL, nw, be, e);
}

int wekuaMatrixSub(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL || b == NULL){
		return CL_INVALID_MEM_OBJECT;
	}else if (a->dtype != b->dtype){
		return CL_INVALID_MEM_OBJECT;
	}

	void *alpha;
	int8_t cone = -1;
	uint8_t dtype = a->dtype;
	int16_t sone = -1;
	int32_t ione = -1;
	int64_t lone = -1;
	float fone = -1.0;
	double done = -1.0;

	if (dtype == 9) alpha = &done;
	else if (dtype == 8) alpha = &fone;
	else if (dtype >= 6) alpha = &lone;
	else if (dtype >= 4) alpha = &ione;
	else if (dtype >= 2) alpha = &sone;
	else alpha = &cone;

	return wekuaBlasAxpy(b, a, alpha, NULL, nw, be, e);
}

int wekuaMatrixDot(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL || b == NULL){
		return CL_INVALID_MEM_OBJECT;
	}else if (a->dtype != b->dtype){
		return CL_INVALID_MEM_OBJECT;
	}else if (memcmp(a->vl_shape, b->vl_shape, 16) != 0){
		return CL_INVALID_MEM_OBJECT;
	}
	uint8_t dtype = a->dtype, com = a->com|b->com;

	wekuaContext ctx = a->ctx;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_DOT, dtype, com);

	if (kernel == NULL) return CL_COMPILE_PROGRAM_FAILURE;

	if (com){
		if (createComplexMatrix(a) || createComplexMatrix(b)){
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 8, &a->vl_shape[1]);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->vl_shape, a->work_items, nw, be, e);
}

int wekuaMatrixDivide(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL && b == NULL){
		return CL_INVALID_MEM_OBJECT;
	}else if (a->dtype != b->dtype){
		return CL_INVALID_MEM_OBJECT;
	}else if (memcmp(a->vl_shape, b->vl_shape, 16) != 0){
		return CL_INVALID_MEM_OBJECT;
	}

	uint8_t dtype = a->dtype, com = a->com|b->com;
	if (dtype < WEKUA_DTYPE_FLOAT) return CL_INVALID_MEM_OBJECT;

	wekuaContext ctx = a->ctx;
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_DIVIDE, dtype, com);

	if (kernel == NULL) return CL_COMPILE_PROGRAM_FAILURE;
	
	if (com){
		if (createComplexMatrix(a)|createComplexMatrix(b)){
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 8, &a->col);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[4], nw, be, e);
}

int wekuaMatrixPower(wmatrix a, wmatrix b, void *exp_r, void *exp_i, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL) return CL_INVALID_MEM_OBJECT;
	else if (a->dtype < WEKUA_DTYPE_FLOAT) return CL_INVALID_MEM_OBJECT;

	uint8_t om = 0, dtype = a->dtype, com = a->com;
	wekuaContext ctx = a->ctx;
	uint32_t dl = ctx->dtype_length[dtype];
	
	cl_mem b_real = NULL, b_imag = NULL;

	if (exp_r == NULL) exp_r = &zero;
	if (exp_i == NULL) exp_i = &zero;

	if (b != NULL){
		om = 1;

		if (memcmp(a->shape, b->shape, 16) != 0) return CL_INVALID_MEM_OBJECT;

		com |= b->com;

		if (com){
			if (createComplexMatrix(a)|createComplexMatrix(b)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
		b_real = b->real;
		b_imag = b->imag;
	}else if (memcmp(&zero, exp_i, dl) != 0){
		if (createComplexMatrix(a)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
	}

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_POWER, dtype, com);
	if (kernel == NULL) return CL_COMPILE_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b_imag);
	clSetKernelArg(kernel, 4, dl, exp_r);
	clSetKernelArg(kernel, 5, dl, exp_i);
	clSetKernelArg(kernel, 6, 8, &a->col);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[4], nw, be, e);
}

int wekuaMatrixLn(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL) return CL_INVALID_MEM_OBJECT;

	uint8_t dtype = a->dtype, com = a->com;
	wekuaContext ctx = a->ctx;
	cl_kernel kernel;

	if (dtype < WEKUA_DTYPE_FLOAT) return CL_INVALID_MEM_OBJECT;

	kernel = compileKernel(ctx, WEKUA_KERNEL_LOG, dtype, com);
	if (kernel == NULL) return CL_COMPILE_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 8, &a->vl_shape[1]);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->vl_shape, a->work_items, nw, be, e);
}

int wekuaMatrixLog(wmatrix a, wmatrix b, void *base_r, void *base_i){
	if (a == NULL && b == NULL){
		return CL_INVALID_MEM_OBJECT;
	}else if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}

	uint8_t dtype = a->dtype, om = 0;
	wekuaContext ctx = a->ctx;
	cl_event e[3];
	int ret = CL_SUCCESS;

	if (dtype < WEKUA_DTYPE_FLOAT){
		return CL_INVALID_MEM_OBJECT;
	}else if (b != NULL){
		if (dtype != b->dtype){
			return CL_INVALID_MEM_OBJECT;
		}
	}else{
		om = 1;
		b = wekuaFillMatrix(ctx, a->shape[0], a->shape[1], base_r, base_i, dtype);
	}

	ret |= wekuaMatrixLn(a, 0, NULL, e);

	if (ret != CL_SUCCESS){
		if (om) wekuaFreeMatrix(b, 0, NULL);
		return ret;
	}

	ret |= wekuaMatrixLn(b, 0, NULL, &e[1]);

	if (ret != CL_SUCCESS){
		if (om) wekuaFreeMatrix(b, 1, e);
		else clWaitForEvents(1, e);
		return ret;
	}

	ret |= wekuaMatrixDivide(a, b, 2, e, &e[3]);

	if (ret == CL_SUCCESS){
		if (om) wekuaFreeMatrix(b, 2, e);
		else clWaitForEvents(3, e);
		for (uint8_t x=0; x<2; x++) clReleaseEvent(e[x]);
	}else{
		if (om) wekuaFreeMatrix(b, 2, e);
		else clWaitForEvents(2, e);
		for (uint8_t x=0; x<2; x++) clReleaseEvent(e[x]);
	}

	return ret;
}

int wekuaMatrixTrace(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be){
	if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	cl_event e;
	int ret;
	wmatrix b;

	b = wekuaMatrixDiag(a, nw, be, &e);
	if (b == NULL){
		return CL_MEM_OBJECT_ALLOCATION_FAILURE;
	}

	ret = wekuaMatrixSum(b, real, imag, 1, &e);
	if (ret == CL_SUCCESS) ret |= clReleaseEvent(e);
	wekuaFreeMatrix(b, 0, NULL);

	return ret;
}

int wekuaMatrixSqrt(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL) return CL_INVALID_MEM_OBJECT;

	uint8_t dtype = a->dtype;
	wekuaContext ctx = a->ctx;
	cl_kernel kernel;

	kernel = compileKernel(ctx, WEKUA_KERNEL_SQRT, dtype, a->com);
	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &a->vl_shape[2], &a->work_items[8], nw, be, e);
}

int wekuaMatrixDet(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be){
	if (a == NULL && real == NULL && imag == NULL) return CL_INVALID_ARG_VALUE;
	else if (a->dtype < WEKUA_DTYPE_FLOAT) return CL_INVALID_MEM_OBJECT;

	wekuaContext ctx = a->ctx;
	uint8_t dtype = a->dtype, com = a->com;
	int ret = CL_SUCCESS;
	uint32_t evn;
	uint64_t col = a->shape[1], rcol = a->vl_shape[1], *shape, *wi;
	void *one = NULL;
	cl_event *e = NULL, *befo = NULL;
	cl_kernel kernel;

	shape = a->vl_shape;
	wi = &a->work_items[3];

	wmatrix b = NULL, c = NULL;

	kernel = compileKernel(ctx, WEKUA_KERNEL_DET, dtype, com);
	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	if (col != a->shape[0]) return CL_INVALID_ARG_VALUE;

	e = (cl_event*) calloc(a->shape[0]+1, sizeof(cl_event));
	if (e == NULL) goto wekua_det_fail;

	one = get_one(dtype, ctx->dtype_length[dtype]);
	if (one == NULL) goto wekua_det_fail;

	b = wekuaMatrixCopy(a, nw, be, e);
	if (b == NULL) goto wekua_det_fail;

	c = wekuaFillMatrix(ctx, a->shape[0], a->shape[1], one, NULL, dtype);
	if (c == NULL) goto wekua_det_fail;

	if (b->com){
		if (createComplexMatrix(c)) goto wekua_det_fail;
	}

	befo = e;
	for (evn=0; evn<col;){
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &c->real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &c->imag);
		clSetKernelArg(kernel, 4, 8, &evn);
		clSetKernelArg(kernel, 5, 8, &rcol);

		ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, shape, wi, 1, befo, &e[evn+1]);
		if (ret != CL_SUCCESS) goto wekua_det_fail;
		befo = &e[evn+1];
		evn++;
	}

	ret = wekuaMatrixMul(c, real, imag, evn, e);
	if (ret == CL_SUCCESS){
		for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);
		free(e);
		e = NULL;
	}

	wekua_det_fail:
	if (one != NULL) free(one);
	if (e != NULL){
		evn++;
		clWaitForEvents(evn, e);
		for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);
		free(e);
	}
	wekuaFreeMatrix(b, 0, NULL);
	wekuaFreeMatrix(c, 0, NULL);
	return ret;
}
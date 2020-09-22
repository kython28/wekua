#include "wekua.h"

void getLWI(uint64_t *x, uint64_t *y, uint32_t si, uint64_t max, uint32_t cu){
	uint64_t c = (uint64_t)(pow(1.0*max, 1.0/si)/cu);
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

void MapBufferMatrix(wmatrix *a){
	if (a == NULL) return;

	if (a->real != NULL && a->raw_real == NULL){
		a->raw_real = clEnqueueMapBuffer(a->ctx->command_queue, a->real, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);	
	}
	if (a->imag != NULL && a->raw_imag == NULL){
		a->raw_imag = clEnqueueMapBuffer(a->ctx->command_queue, a->imag, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);
	}
}

void UnmapBufferMatrix(wmatrix *a){
	if (a == NULL) return;

	if (a->real != NULL && a->raw_real != NULL){
		clEnqueueUnmapMemObject(a->ctx->command_queue, a->real, a->raw_real, 0, NULL, NULL);
		a->raw_real = NULL;
	}
	if (a->imag != NULL && a->raw_imag != NULL){
		clEnqueueUnmapMemObject(a->ctx->command_queue, a->imag, a->raw_imag, 0, NULL, NULL);
		a->raw_imag = NULL;
	}
}

uint8_t createComplexMatrix(wmatrix *b){
	if (b == NULL){
		return 1;
	}else if (b->com){
		return 0;
	}
	wekuaContext *ctx = b->ctx;
	uint64_t size = b->size;
	int ret;

	b->imag = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(double)*size, NULL, &ret);
	if (ret != CL_SUCCESS){
		return 1;
	}
	MapBufferMatrix(b);
	cl_event ie;
	double beta = 0.0;
	clEnqueueFillBuffer(ctx->command_queue, b->imag, &beta, sizeof(double), 0, size*sizeof(double), 0, NULL, &ie);
	clWaitForEvents(1, &ie);
	clReleaseEvent(ie);
	b->com = 1;
	if (b->parent != NULL){
		((wmatrix*)b->parent)->com = 1;
		((wmatrix*)b->parent)->imag = b->imag;
		((wmatrix*)b->parent)->raw_imag = b->raw_imag;
	}

	return 0;
}

void removeComplexMatrix(wmatrix *b, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (b == NULL){
		return;
	}else if (b->com == 0){
		return;
	}
	cl_event e;
	clEnqueueUnmapMemObject(b->ctx->command_queue, b->imag, b->raw_imag, 0, NULL, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	b->raw_imag = NULL;

	clReleaseMemObject(b->imag);
	b->imag = NULL;
	b->com = 0;
	if (b->parent != NULL){
		((wmatrix*)b->parent)->com = 0;
		((wmatrix*)b->parent)->imag = NULL;
		((wmatrix*)b->parent)->raw_imag = NULL;
	}
	return;
}

void wekuaFreeMatrix(wmatrix *a, uint32_t nw, cl_event *be){
	if (a == NULL){
		return;
	}else if (a->sm){
		free(a);
		return;
	}
	clWaitForEvents(nw, be);
	UnmapBufferMatrix(a);
	if (a->real != NULL){
		clReleaseMemObject(a->real);
	}
	if (a->imag != NULL){
		clReleaseMemObject(a->imag);
	}
	free(a);
}

void wekuaMatrixRealPrint(wmatrix *a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	double real;
	for (uint64_t y=0; y<a->shape[0]; y++){
		for (uint64_t x=0; x<a->shape[1]; x++){
			if (x == 0 && (y < 5 || y >= a->shape[0]-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= a->shape[1]-4) && (y < 4 || y >= a->shape[0]-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%14.5e", real);
				if (y+1 != a->shape[0] || x+1 != a->shape[1]){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= a->shape[0]-4)) || (y == 4 && (x < 4 || x >= a->shape[1]-4))){
				printf("%15s", "... ");
			}
			if (a->shape[0] > 1 && x == a->shape[1]-1 && (y < 5 || y >= a->shape[0]-4)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrint(wmatrix *a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[21];
	double real, imag;
	for (uint32_t y=0; y<a->shape[0]; y++){
		for (uint32_t x=0; x<a->shape[1]; x++){
			if (x == 0 && (y < 3 || y >= a->shape[0]-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= a->shape[1]-2) && (y < 2 || y >= a->shape[0]-2)){
				memset(num, 0, 21);
				wekuaGetValueFromMatrix(a, y, x, &real, &imag, 0, NULL);
				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%.2e%+.2ej", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%.5e", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%.5ej", imag);
				}else{
					sprintf(num, "%.5e", real);
				}
				printf("%24s", num);
				if (y+1 != a->shape[0] || x+1 != a->shape[1]){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= a->shape[0]-2)) || (y == 2 && (x < 3 || x >= a->shape[1]-2))){
				printf("%25s", "... ");
			}
			if (a->shape[0] > 1 && x == a->shape[1]-1 && (y < 3 || y >= a->shape[0]-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixPrint(wmatrix *a, uint32_t nw, cl_event *e){
	if (a == NULL){
		return;
	}
	clWaitForEvents(nw, e);
	printf("wmatrix([");
	if(a->com){
		wekuaMatrixComplexPrint(a);
	}else{
		wekuaMatrixRealPrint(a);
	}
	printf("], shape=(%ld, %ld))\n", a->shape[0], a->shape[1]);
}

wmatrix *wekuaAllocMatrix(wekuaContext *ctx, uint64_t r, uint64_t c){
	if (ctx == NULL || r == 0 || c == 0){
		return NULL;
	}
	wmatrix *a = (wmatrix*) malloc(sizeof(wmatrix));
	uint64_t max = ctx->max_work_group_size;
	uint32_t cu = ctx->compute_units;
	memset(a->offset, 0, 16);
	a->com = 0;
	a->sm = 0;
	a->parent = NULL;

	a->shape[0] = r;
	a->shape[1] = c;

	a->size = r*c;
	a->real_size[0] = r;
	a->real_size[1] = c;

	a->ctx = ctx;
	getLWI(&a->size, a->work_items, 1, max, cu);
	getLWI(a->shape, &a->work_items[3], 1, max, cu);
	getLWI(&a->shape[1], &a->work_items[4], 1, max, cu);
	getLWI(a->shape, &a->work_items[1], 2, max, cu);

	int ret;
	a->real = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(double)*a->size, NULL, &ret);
	if (ret != 0){
		printf("Failed to allocate new memory :-(\n");
		free(a);
		return NULL;
	}
	a->raw_real = NULL;

	a->imag = NULL;
	a->raw_imag = NULL;

	MapBufferMatrix(a);
	return a;
}

wmatrix *wekuaAllocComplexMatrix(wekuaContext *ctx, uint64_t r, uint64_t c){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	if (createComplexMatrix(a)){
		wekuaFreeMatrix(a, 0, NULL);
		return NULL;
	}
	return a;
}

wmatrix *wekuaFillMatrix(wekuaContext *ctx, uint64_t r, uint64_t c, double alpha, double beta){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	if (a == NULL){
		return NULL;
	}
	if (beta != 0.0){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
	}
	cl_event e;
	clEnqueueFillBuffer(a->ctx->command_queue, a->real, &alpha, sizeof(double), 0, a->size*sizeof(double), 0, NULL, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);
	if (beta != 0.0){
		clEnqueueFillBuffer(a->ctx->command_queue, a->imag, &beta, sizeof(double), 0, a->size*sizeof(double), 0, NULL, &e);
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}
	return a;
}

wmatrix *wekuaMatrixRandn(wekuaContext *ctx, uint64_t r, uint64_t c, uint8_t com){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	if (a == NULL){
		return NULL;
	}
	if (com){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
	}

	cl_kernel kernel = ctx->kernels[0];

	cl_mem ran_r=NULL, ran_i=NULL;
	uint64_t *ran_r_m, *ran_i_m;
	int ret;
	cl_event e;
	
	ran_r = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, a->size*8, NULL, &ret);
	if (ret != 0){
		clReleaseMemObject(ran_r);
		return NULL;
	}
	ran_r_m = clEnqueueMapBuffer(ctx->command_queue, ran_r, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*8, 0, 0, NULL, NULL);
	getRandomBuffer(ran_r_m, a->size*8);
	clEnqueueUnmapMemObject(ctx->command_queue, ran_r, ran_r_m, 0, NULL, NULL);

	if (a->com){
		ran_i = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, a->size*8, NULL, &ret);
		if (ret != 0){
			clReleaseMemObject(ran_r);
			clReleaseMemObject(ran_i);
			return NULL;
		}
		ran_i_m = clEnqueueMapBuffer(ctx->command_queue, ran_i, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*8, 0, 0, NULL, NULL);
		getRandomBuffer(ran_i_m, a->size*8);
		clEnqueueUnmapMemObject(ctx->command_queue, ran_i, ran_i_m, 0, NULL, NULL);
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &ran_r);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &ran_i);
	clSetKernelArg(kernel, 4, 1, &a->com);
	
	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &a->size, a->work_items, 0, NULL, &e);

	clWaitForEvents(1, &e);

	clReleaseMemObject(ran_r);
	clReleaseMemObject(ran_i);

	clReleaseEvent(e);

	return a;
}

wmatrix *wekuaMatrixRandUniform(wekuaContext *ctx, uint64_t r, uint64_t c, double ra, double ia, double re, double ie, uint8_t com){
	if (ctx == NULL){
		return NULL;
	}
	cl_kernel kernel = ctx->kernels[19];
	cl_event e;
	wmatrix *a = wekuaMatrixRandn(ctx, r, c, com);

	if (com){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(double), &ra);
	clSetKernelArg(kernel, 3, sizeof(double), &ia);
	clSetKernelArg(kernel, 4, sizeof(double), &re);
	clSetKernelArg(kernel, 5, sizeof(double), &ie);
	clSetKernelArg(kernel, 6, 8, &a->shape[1]);
	clSetKernelArg(kernel, 7, 1, &com);

	clEnqueueNDRangeKernel(a->ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1], 0, NULL, &e);

	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	return a;
}

wmatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint64_t r, uint64_t c, void *rbuf, void *ibuf){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	memcpy(a->raw_real, rbuf, a->size*sizeof(double));
	if (ibuf != NULL){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
		memcpy(a->raw_imag, ibuf, a->size*sizeof(double));
	}
	return a;
}

wmatrix *wekuaMatrixCopy(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return NULL;
	}

	wmatrix *b; double alpha=0.0;
	uint64_t size;

	wekuaContext *ctx = a->ctx;
	b = (wmatrix*) malloc(sizeof(wmatrix));
	b->com = 0;
	b->sm = 0;
	b->ctx = ctx;
	b->parent = NULL;

	memcpy(b->shape, a->shape, 16);
	memcpy(b->real_size, a->shape, 16);
	memcpy(b->work_items, a->work_items, 40);
	memset(b->offset, 0, 16);
	
	size = a->shape[0]*a->shape[1];
	b->size = size;

	int ret;
	b->real = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(double)*size, NULL, &ret);
	if (ret != 0){
		printf("Failed to allocate new memory :-(\n");
		free(b);
		return NULL;
	}
	b->imag = NULL;
	b->raw_real = NULL;
	b->raw_imag = NULL;

	cl_event ie;
	ret = clEnqueueFillBuffer(ctx->command_queue, b->real, &alpha, sizeof(double), 0, size*sizeof(double), 0, NULL, &ie);
	if (ret != 0){
		clReleaseMemObject(b->real);
		free(b);
		return NULL;
	}

	MapBufferMatrix(b);
	wekuaMatrixAdd(b, a, 1, &ie, e);

	clWaitForEvents(1, &ie);
	clReleaseEvent(ie);

	return b;
}

wmatrix *wekuaCutMatrix(wmatrix *a, uint64_t x, uint64_t w, uint64_t y, uint64_t h){
	if (a == NULL){
		return NULL;
	}else if (w == 0 || h == 0 || x+w > a->shape[1] || y+h > a->shape[0]){
		return NULL;
	}
	wmatrix *b = (wmatrix*) malloc(sizeof(wmatrix));
	wekuaContext *ctx = a->ctx;
	uint64_t max = ctx->max_work_group_size;
	uint32_t cu = ctx->compute_units;

	b->ctx = a->ctx;
	b->size = a->size;
	b->parent = a;

	b->com = a->com;
	b->sm = 1;

	b->shape[0] = h;
	b->shape[1] = w;

	memcpy(b->offset, a->offset, 16);
	b->offset[0] += y;
	b->offset[1] += x;

	memcpy(b->real_size, a->real_size, 16);

	b->real = a->real;
	b->imag = a->imag;

	b->raw_real = a->raw_real;
	b->raw_imag = a->raw_imag;

	b->work_items[0] = a->work_items[0];

	getLWI(b->shape, &b->work_items[3], 1, max, cu);
	getLWI(&b->shape[1], &b->work_items[4], 1, max, cu);
	getLWI(b->shape, &b->work_items[1], 2, max, cu);

	return b;
}

wmatrix *wekuaMatrixResize(wmatrix *a, uint64_t r, uint64_t c, double alpha, double beta, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return NULL;
	}else if (r == 0 || c == 0){
		return NULL;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[15];

	wmatrix *b = wekuaAllocMatrix(ctx, r, c);
	if (a->com){
		if (createComplexMatrix(b) || createComplexMatrix(a)){
			wekuaFreeMatrix(b, 0, NULL);
			return NULL;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 4, 8, &b->real_size[1]);
	clSetKernelArg(kernel, 5, 8, b->real_size);
	clSetKernelArg(kernel, 6, 8, &a->shape[1]);
	clSetKernelArg(kernel, 7, 8, a->shape);
	clSetKernelArg(kernel, 8, 8, &a->real_size[1]);
	clSetKernelArg(kernel, 9, 8, a->offset);
	clSetKernelArg(kernel, 10, 8, &a->offset[1]);
	clSetKernelArg(kernel, 11, sizeof(double), &alpha);
	clSetKernelArg(kernel, 12, sizeof(double), &beta);
	clSetKernelArg(kernel, 13, 1, &b->com);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, b->shape, &b->work_items[1], nw, be, e);
	return b;
}

void wekuaReshapeMatrix(wmatrix *a, uint64_t r, uint64_t c, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (a == NULL){
		return;
	}else if (a->sm){
		return;
	}else if (a->shape[0]*a->shape[1] != r*c){
		return;
	}

	a->shape[0] = r;
	a->shape[1] = c;

	a->real_size[0] = r;
	a->real_size[1] = c;
}

void wekuaGetValueFromMatrix(wmatrix *a, uint64_t y, uint64_t x, double *real, double *imag, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (a == NULL){
		return;
	}else if (y > a->shape[0]){
		return;
	}else if (x > a->shape[1]){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}
	uint64_t *offset = a->offset, col = a->real_size[1];
	uint64_t posi = (offset[0]+y)*col+x+offset[1];
	if (real != NULL){
		real[0] = a->raw_real[posi];
	}
	if (imag != NULL && a->com){
		imag[0] = a->raw_imag[posi];
	}
}

void wekuaPutValueToMatrix(wmatrix *a, uint64_t y, uint64_t x, double real, double imag, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (a == NULL){
		return;
	}else if (y > a->shape[0]){
		return;
	}else if (x > a->shape[1]){
		return;
	}
	uint64_t *offset = a->offset, col = a->real_size[1];
	uint64_t posi = (offset[0]+y)*col+x+offset[1];

	a->raw_real[posi] = real;
	if (imag != 0.0){
		if (a->com == 0){
			if (createComplexMatrix(a)){
				return;
			}
		}
		a->raw_imag[posi] = imag;
	}
}

// Basic functions

wmatrix *wekuaMatrixIden(wekuaContext *ctx, uint64_t c){
	cl_event e;
	cl_kernel kernel = ctx->kernels[1];
	wmatrix *i = wekuaFillMatrix(ctx, c, c, 0.0, 0.0);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &i->real);
	clSetKernelArg(kernel, 1, 8, &i->shape[0]);
	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, i->shape, &i->work_items[3], 0, NULL, &e);

	clWaitForEvents(1, &e);
	clReleaseEvent(e);


	return i;
}

wmatrix *wekuaMatrixTrans(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return NULL;
	}
	wekuaContext *ctx = a->ctx;
	wmatrix *b;
	if (a->com){
		b = wekuaAllocComplexMatrix(ctx, a->shape[1], a->shape[0]);
	}else{
		b = wekuaAllocMatrix(ctx, a->shape[1], a->shape[0]);
	}

	cl_kernel kernel = ctx->kernels[2];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 4, 8, &a->real_size[1]);
	clSetKernelArg(kernel, 5, 8, a->shape);
	clSetKernelArg(kernel, 6, 1, &a->com);
	clSetKernelArg(kernel, 7, 8, a->offset);
	clSetKernelArg(kernel, 8, 8, &a->offset[1]);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1], nw, be, e);

	return b;
}

wmatrix *wekuaMatrixProduct(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL || b == NULL){
		return NULL;
	}else if (a->shape[1] != b->shape[0]){
		return NULL;
	}
	wmatrix *c = wekuaFillMatrix(a->ctx, a->shape[0], b->shape[1], 0.0, 0.0);
	wekuaBlasGemm(1.0, 0.0, 0, a, 0, b, 0.0, 0.0, c, nw, be, e);
	return c;
}

wmatrix *wekuaMatrixDiag(wmatrix *a, uint32_t nw, cl_event *be){
	if (a == NULL){
		return NULL;
	}
	cl_event e;
	wmatrix *b, *c;
	uint8_t typ = 1;
	wekuaContext *ctx = a->ctx;
	uint64_t dim, wi;
	dim = a->shape[0];
	wi = a->work_items[3];

	cl_kernel kernel = ctx->kernels[13];
	c = wekuaMatrixCopy(a, nw, be, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	if (a->shape[0] == 1 || a->shape[1] == 1){
		if (dim < a->shape[1]){
			dim = a->shape[1];
			wi = a->work_items[4];
		}
		b = wekuaFillMatrix(ctx, dim, dim, 0.0, 0.0);
	}else if (a->shape[0] == a->shape[1]){
		b = wekuaFillMatrix(ctx, 1, dim, 0.0, 0.0);
		typ = 0;
	}else{
		return NULL;
	}

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b, 0, NULL);
			return NULL;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &c->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &c->imag);
	clSetKernelArg(kernel, 4, 1, &typ);
	clSetKernelArg(kernel, 5, 1, &a->com);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &dim, &wi, nw, be, &e);

	wekuaFreeMatrix(c, 1, &e);
	clReleaseEvent(e);

	return b;
}

wmatrix *wekuaArange(wekuaContext *ctx, double x, double y, double alpha, uint8_t trans){
	wmatrix *a;
	int64_t col = fabs((y-x)/alpha);
	while (x+col*alpha > y && col > 0){
		col--;
	}

	cl_event e;
	cl_kernel kernel = ctx->kernels[27];
	if (trans){
		a = wekuaFillMatrix(ctx, (uint64_t)col, 1, x, 0.0);
	}else{
		a = wekuaFillMatrix(ctx, 1, (uint64_t)col, x, 0.0);
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(double), &alpha);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &a->size, a->work_items, 0, NULL, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	return a;
}

wmatrix *wekuaMatrixAbs(wmatrix *a, uint32_t nw, cl_event *be){
	if (a == NULL){
		return NULL;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[12];
	cl_event ie, ae;
	wmatrix *b = wekuaMatrixCopy(a, nw, be, &ie);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, 1, &a->com);
	clSetKernelArg(kernel, 3, 8, &a->real_size[1]);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1], 1, &ie, &ae);

	removeComplexMatrix(b, 1, &ae);

	clReleaseEvent(ie);
	clReleaseEvent(ae);

	return b;
}

void wekuaMatrixAdd(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e){
	if (memcmp(a->shape, b->shape, 16) != 0){
		return;
	}
	wekuaBlasAxpy(1.0, 0.0, b, a, nw, be, e);
}

void wekuaMatrixSub(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e){
	if (memcmp(a->shape, b->shape, 16) != 0){
		return;
	}
	wekuaBlasAxpy(-1.0, 0.0, b, a, nw, be, e);
}

void wekuaMatrixDotScalar(wmatrix *a, double alpha, double beta, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[11];

	if (beta != 0.0 && a->com == 0){
		if (createComplexMatrix(a)){
			return;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 1, &a->com);
	clSetKernelArg(kernel, 3, sizeof(double), &alpha);
	clSetKernelArg(kernel, 4, sizeof(double), &beta);
	clSetKernelArg(kernel, 5, 8, &a->real_size[1]);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1], nw, be, e);
}

void wekuaMatrixDot(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL || b == NULL){
		return;
	}else if (a->com || b->com){
		if (createComplexMatrix(a) || createComplexMatrix(b)){
			return;
		}
	}else if (memcmp(a->shape, b->shape, 16) != 0){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[24];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 1, &a->com);
	clSetKernelArg(kernel, 5, 8, &a->real_size[1]);
	clSetKernelArg(kernel, 6, 8, &b->real_size[1]);
	clSetKernelArg(kernel, 7, 8, a->offset);
	clSetKernelArg(kernel, 8, 8, &a->offset[1]);
	clSetKernelArg(kernel, 9, 8, b->offset);
	clSetKernelArg(kernel, 10, 8, &b->offset[1]);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1], nw, be, e);
}

void wekuaMatrixAbsdiff(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be){
	if (a == NULL || b == NULL){
		return;
	}
	cl_event ie;
	wekuaMatrixSub(a, b, nw, be, &ie);
	wekuaMatrixAbs(a, 1, &ie);

	clWaitForEvents(1, &ie);
	clReleaseEvent(ie);
}

void wekuaMatrixLn(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[23];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 1, &a->com);
	clSetKernelArg(kernel, 3, 8, &a->real_size[1]);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1], nw, be, e);
}

void wekuaMatrixLog(wmatrix *a, double r_base, double i_base, uint32_t nw, cl_event *be){
	if (a == NULL){
		return;
	}
	cl_event ie[3];
	wmatrix *b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], r_base, i_base);
	wekuaMatrixLn(a, nw, be, ie);
	wekuaMatrixLn(b, 1, ie, &ie[1]);
	wekuaMatrixDivide(a, b, 1, &ie[1], &ie[2]);
	wekuaFreeMatrix(b, 1, &ie[2]);

	clWaitForEvents(1, &ie[2]);
	for (uint32_t i=0; i<3; i++) clReleaseEvent(ie[i]);
}

void wekuaMatrixDivide(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL || b == NULL){
		return;
	}else if (memcmp(a->shape, b->shape, 16)){
		return;
	}else if (a->com || b->com){
		if (createComplexMatrix(a) || createComplexMatrix(b)){
			return;
		}
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[25];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 1, &a->com);
	clSetKernelArg(kernel, 5, 8, &a->real_size[1]);
	clSetKernelArg(kernel, 6, 8, &b->real_size[1]);
	clSetKernelArg(kernel, 7, 8, a->offset);
	clSetKernelArg(kernel, 8, 8, &a->offset[1]);
	clSetKernelArg(kernel, 9, 8, b->offset);
	clSetKernelArg(kernel, 10, 8, &b->offset[1]);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1], nw, be, e);
}

void wekuaMatrixPowr(wmatrix *a, double real, double imag, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[26];

	if (a->com == 0 && imag != 0.0){
		if (createComplexMatrix(a)){
			return;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(double), &real);
	clSetKernelArg(kernel, 3, sizeof(double), &imag);
	clSetKernelArg(kernel, 4, 1, &a->com);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1], nw, be, e);
}

// Extra functions

void wekuaMatrixSum(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}
	cl_event events[2];

	wmatrix *b, *c, *d, *e;
	b = wekuaFillMatrix(a->ctx, a->shape[1], 1, 1.0, 0.0);
	d = wekuaFillMatrix(a->ctx, 1, a->shape[0], 1.0, 0.0);
	
	c = wekuaMatrixProduct(a, b, nw, be, events);
	wekuaFreeMatrix(b, 1, events);

	e = wekuaMatrixProduct(d, c, 0, NULL, &events[1]);
	wekuaFreeMatrix(c, 1, &events[1]);
	wekuaFreeMatrix(d, 0, NULL);

	if (real != NULL){
		real[0] = e->raw_real[0];
	}
	if (a->com && imag != NULL){
		imag[0] = e->raw_imag[0];
	}
	wekuaFreeMatrix(e, 0, NULL);

	for (uint32_t x=0; x<2; x++) clReleaseEvent(events[x]);
}

void wekuaMatrixMul(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}
	cl_event ie;

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[14];

	double re=1.0, im=1.0, aa, bb;
	uint64_t col = a->real_size[1];

	wmatrix *b;

	if (a->shape[1] == 1 || a->shape[0] == 1){
		b = wekuaMatrixCopy(a, nw, be, &ie);
		goto Sumstetwo;
	}

	b = wekuaFillMatrix(ctx, 1, a->shape[0], 1.0, 0.0);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b, nw, be);
			return;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 8, &col);
	clSetKernelArg(kernel, 5, 1, &a->com);
	clSetKernelArg(kernel, 6, 8, a->offset);
	clSetKernelArg(kernel, 7, 8, &a->offset[1]);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, a->shape, &a->work_items[3], nw, be, &ie);

	Sumstetwo:

	clWaitForEvents(1, &ie);
	clReleaseEvent(ie);

	if (a->com){
		for (uint64_t i=0; i<b->shape[0]; i++){
			for (uint64_t j=0; i<b->shape[1]; i++){
				aa = b->raw_real[i*col+j];
				bb = b->raw_imag[i*col+j];
				re = aa*re - bb*im;
				im = aa*im + bb*re;
			}
		}
			
	}else{
		for (uint64_t i=0; i<b->shape[0]; i++){
			for (uint64_t j=0; i<b->shape[1]; i++){
				re *= b->raw_real[i*col+j];
			}
		}
	}

	if (real != NULL){
		real[0] = re;
	}
	if (a->com && imag != NULL){
		imag[0] = im;
	}

	wekuaFreeMatrix(b, 0, NULL);

}

void wekuaMatrixMean(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be){
	cl_event e[2];

	wmatrix *b = wekuaMatrixCopy(a, nw, be, e);
	uint64_t *shape = a->shape;
	double n = 1.0/(shape[0]*shape[1]);
	wekuaMatrixDotScalar(b, n, 0.0, 1, e, &e[1]);
	wekuaMatrixSum(b, real, imag, 1, &e[1]);

	clReleaseEvent(e[0]);
	clReleaseEvent(e[1]);

	wekuaFreeMatrix(b, 0, NULL);
}

void wekuaMatrixDet(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}else if (a->shape[0] != a->shape[1]){
		return;
	}else if (a->sm){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[16];
	cl_command_queue cmd = ctx->command_queue;
	cl_event ie;

	wmatrix *b = wekuaMatrixCopy(a, nw, be, &ie);
	wmatrix *c = wekuaFillMatrix(ctx, a->shape[0], a->shape[1], 1.0, 0.0);

	clWaitForEvents(1, &ie);
	clReleaseEvent(ie);

	if (b->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(c, 0, NULL);
			wekuaFreeMatrix(b, 0, NULL);
			return;
		}
	}

	uint64_t col = b->shape[1];
	cl_event *event = (cl_event*) calloc(col-1, sizeof(cl_event));
	cl_event *befo = NULL;
	uint64_t we=0;

	for (uint64_t k=0; k < a->shape[1]-1; k++){
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &c->real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &c->imag);
		clSetKernelArg(kernel, 4, 8, &k);
		clSetKernelArg(kernel, 5, 8, &col);
		clSetKernelArg(kernel, 6, 1, &b->com);

		clEnqueueNDRangeKernel(cmd, kernel, 1, NULL, a->shape, &a->work_items[3], we, befo, &event[k]);
		if (we == 0){
			we++;
		}
		befo = &event[k];
	}

	clWaitForEvents(1, &event[a->shape[1]-2]);
	for (uint64_t x=0; x < a->shape[1]-1; x++){
		clReleaseEvent(event[x]);
	}
	free(event);

	wekuaMatrixMul(c, real, imag, 0, NULL);

	wekuaFreeMatrix(b, 0, NULL);
	wekuaFreeMatrix(c, 0, NULL);
}

wmatrix *wekuaMatrixInv(wmatrix *a, uint32_t nw, cl_event *be){
	if (a == NULL){
		return NULL;
	}else if (a->shape[0] != a->shape[1]){
		return NULL;
	}else if (a->sm){
		return NULL;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[17], kernel2 = ctx->kernels[18];
	cl_command_queue cmd = ctx->command_queue;
	cl_event ie;

	wmatrix *inv, *b;
	b = wekuaMatrixCopy(a, nw, be, &ie);
	inv = wekuaMatrixIden(a->ctx, a->shape[0]);

	clWaitForEvents(1, &ie);
	clReleaseEvent(ie);

	if (a->com){
		if (createComplexMatrix(inv)){
			wekuaFreeMatrix(inv, 0, NULL);
			return NULL;
		}
	}


	cl_event *event, *befo = NULL;
	uint32_t we=0;
	uint64_t col = a->shape[1];
	int ret;

	event = (cl_event*) calloc((2*(col-1)+1), sizeof(cl_event));

	uint8_t otherm=1, com = a->com;

	for (uint8_t t=0; t<2; t++){
		for (uint64_t k=0; k < col-1; k++){
			
			clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &inv->real);
			clSetKernelArg(kernel, 3, sizeof(cl_mem), &inv->imag);
			clSetKernelArg(kernel, 4, 8, &k);
			clSetKernelArg(kernel, 5, 8, &col);
			clSetKernelArg(kernel, 6, 1, &com);
			clSetKernelArg(kernel, 7, 1, &otherm);
			clSetKernelArg(kernel, 8, 1, &t);

			ret = clEnqueueNDRangeKernel(cmd, kernel, 1, NULL, a->shape, &a->work_items[3], we, befo, &event[t*(col-1)+k]);
			if (ret != 0){
				printf("Failed to run kernel %i :-(\n", ret);
				exit(1);
			}else if (we == 0){
				we++;
			}
			befo = &event[t*(col-1)+k];
		}
	}
	clSetKernelArg(kernel2, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel2, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel2, 2, sizeof(cl_mem), &inv->real);
	clSetKernelArg(kernel2, 3, sizeof(cl_mem), &inv->imag);
	clSetKernelArg(kernel2, 4, 8, &col);
	clSetKernelArg(kernel2, 5, 1, &a->com);

	clEnqueueNDRangeKernel(cmd, kernel2, 2, NULL, a->shape, &a->work_items[1], 1, befo, &event[2*(col-1)]);

	clWaitForEvents(1, &event[2*(col-1)]);
	for (uint64_t x=0; x < 2*(col-1)+1; x++){
		clReleaseEvent(event[x]);
	}
	free(event);
	wekuaFreeMatrix(b, 0, NULL);
	return inv;
}

wmatrix *wekuaMatrixSolve(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be){
	cl_event e;

	wmatrix *c = wekuaMatrixInv(a, nw, be);
	wmatrix *d = wekuaMatrixProduct(c, b, 0, NULL, &e);
	wekuaFreeMatrix(c, 1, &e);
	clReleaseEvent(e);
	return d;
}

uint32_t wekuaMatrixRang(wmatrix *a, uint32_t nw, cl_event *be){
	if (a == NULL){
		return 0;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[17];
	cl_command_queue cmd = ctx->command_queue;

	uint32_t rang=0;
	wmatrix *b, *c, *e, *f;
	e = wekuaFillMatrix(a->ctx, a->shape[1], 1, 0.0, 0.0);

	cl_mem nullptr = NULL;
	uint64_t wi[2];
	wi[0] = a->work_items[3];
	wi[1] = 1;

	uint8_t d = 0;
	if (a->shape[1] > a->shape[0]){
		d = 1;
	}

	cl_event *event = (cl_event*) malloc(sizeof(cl_event)*a->shape[1]);
	cl_event *befo;
	uint32_t col = a->real_size[1];

	b = wekuaMatrixCopy(a, nw, be, &event[a->shape[1]-1]);
	befo = &event[a->shape[1]-1];

	uint8_t otherm=0, t=0;

	for (uint64_t k=0; k < a->shape[1]-1; k++){
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &nullptr);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &nullptr);
		clSetKernelArg(kernel, 4, 8, &k);
		clSetKernelArg(kernel, 5, 8, &col);
		clSetKernelArg(kernel, 6, 1, &a->com);
		clSetKernelArg(kernel, 7, 1, &otherm);
		clSetKernelArg(kernel, 8, 1, &t);

		clEnqueueNDRangeKernel(cmd, kernel, 1, NULL, &a->shape[d], &a->work_items[3+d], 1, befo, &event[k]);
		befo = &event[k];
	}

	
	
	f = wekuaMatrixAbs(b, 1, &event[a->shape[1]-2]);
	wekuaFreeMatrix(b, 0, NULL);

	clReleaseEvent(event[0]);

	c = wekuaMatrixProduct(f, e, 0, NULL, event);
	wekuaFreeMatrix(c, 1, event);

	for (uint64_t r=0; r < c->size; r++){
		if (c->raw_real[r] > CL_FLT_EPSILON){
			rang++;
		}
	}
	wekuaFreeMatrix(b, 0, NULL);
	wekuaFreeMatrix(c, 0, NULL);

	for (uint64_t x=0; x < a->shape[1]; x++){
		clReleaseEvent(event[x]);
	}
	free(event);
	
	return rang;
}

wmatrix *wekuaMatrixPinv(wmatrix *a, uint32_t nw, cl_event *be){
	if (a == NULL){
		return NULL;
	}
	cl_event e;
	wmatrix *pinv=NULL, *ta=NULL, *te=NULL, *ti=NULL;
	
	ta = wekuaMatrixTrans(a, nw, be, &e);
	uint32_t rang = wekuaMatrixRang(a, 1, &e);

	clReleaseEvent(e);

	if (rang == a->shape[0]){

		te = wekuaMatrixProduct(a, ta, 0, NULL, &e);
		ti = wekuaMatrixInv(te, 1, &e);
		clReleaseEvent(e);

		pinv = wekuaMatrixProduct(ta, ti, 0, NULL, &e);

	}else if (rang == a->shape[1]){

		te = wekuaMatrixProduct(ta, a, 0, NULL, &e);
		ti = wekuaMatrixInv(te, 1, &e);
		clReleaseEvent(e);

		pinv = wekuaMatrixProduct(ti, ta, 0, NULL, &e);

	}
	wekuaFreeMatrix(ta, 1, &e);
	wekuaFreeMatrix(te, 0, NULL);
	wekuaFreeMatrix(ti, 0, NULL);

	clReleaseEvent(e);

	return pinv;
}

wmatrix *wekuaComplexToMatrix(wekuaContext *ctx, double r, double i){
	if (ctx == NULL){
		return NULL;
	}
	wmatrix *a = wekuaAllocMatrix(ctx, 2, 2);
	a->raw_real[0] = r;
	a->raw_real[1] = -1.0*i;
	a->raw_real[2] = i;
	a->raw_real[3] = r;
	return a;
}

void wekuaMatrixMax(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (a == NULL || (real == NULL && imag == NULL)){
		return;
	}

	uint64_t *shape = a->shape;
	uint8_t com = a->com;
	double r, i;
	double rt = CL_DBL_MIN, it = CL_DBL_MIN;

	if (com){
		for (uint64_t y=0; y<shape[0]; y++){
			for (uint64_t x=0; x < shape[1]; x++){
				wekuaGetValueFromMatrix(a, y, x, &r, &i, 0, NULL);
				if (sqrt(r*r + i*i) > sqrt(rt*rt + it*it)){
					rt = r; it = i;
				}
			}
		}
	}else if (com == 0){
		for (uint64_t y=0; y< shape[0]; y++){
			for (uint64_t x=0; x < shape[1]; x++){
				wekuaGetValueFromMatrix(a, y, x, &r, NULL, 0, NULL);
				if (r > rt){
					rt = r;
				}
			}
		}
	}
	if (real != NULL){
		real[0] = rt;
	}
	if (imag != NULL && com){
		imag[0] = it;
	}
}

void wekuaMatrixMin(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (a == NULL || (real == NULL && imag == NULL)){
		return;
	}

	uint64_t *shape = a->shape;
	uint8_t com = a->com;
	double r, i;
	double rt = CL_DBL_MAX, it = CL_DBL_MAX;

	if (com){
		for (uint64_t y=0; y<shape[0]; y++){
			for (uint64_t x=0; x < shape[1]; x++){
				wekuaGetValueFromMatrix(a, y, x, &r, &i, 0, NULL);
				if (sqrt(r*r + i*i) < sqrt(rt*rt + it*it)){
					rt = r; it = i;
				}
			}
		}
	}else if (com == 0){
		for (uint64_t y=0; y< shape[0]; y++){
			for (uint64_t x=0; x < shape[1]; x++){
				wekuaGetValueFromMatrix(a, y, x, &r, NULL, 0, NULL);
				if (r < rt){
					rt = r;
				}
			}
		}
	}
	if (real != NULL){
		real[0] = rt;
	}
	if (imag != NULL && com){
		imag[0] = it;
	}
}

void wekuaMatrixToComplex(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);

	if (a == NULL){
		return;
	}else if (a->shape[1] != 2 || a->shape[0] != 2){
		return;
	}
	if (real != NULL){
		real[0] = a->raw_real[0];
	}
	if (imag != NULL){
		imag[0] = a->raw_real[2];
	}
}

wmatrix *wekuaComplexRandomToMatrix(wekuaContext *ctx){
	if (ctx == NULL){
		return NULL;
	}
	wmatrix *a, *b;
	a = wekuaMatrixRandn(ctx, 2, 1, 0);
	b = wekuaComplexToMatrix(ctx, a->raw_real[0], a->raw_real[1]);
	wekuaFreeMatrix(a, 0, NULL);
	return b;
}

void wekuaMatrixTrace(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be){
	wmatrix *b = wekuaMatrixDiag(a, nw, be);
	wekuaMatrixSum(b, real, imag, 0, NULL);
}

wmatrix *wekuaMatrixPoly(wmatrix *a){
	if (a == NULL){
		return NULL;
	}else if (a->shape[1] != a->shape[0]){
		return NULL;
	}
	wmatrix *c, **b, *i;
	c = wekuaAllocMatrix(a->ctx, 1, a->shape[0]+1);
	if (a->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(c, 0, NULL);
			return NULL;
		}
	}
	b = (wmatrix**) malloc(sizeof(wmatrix*)*2);
	b[0] = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 0.0, 0.0);
	uint64_t r = a->shape[0];
	c->raw_real[r] = 1.0;
	cl_event e[3];

	for (uint64_t x=1; x<=r; x++){
		i = wekuaMatrixIden(a->ctx, r);
		b[1] = wekuaMatrixProduct(a, b[0], 0, NULL, e);

		if (a->com){
			wekuaMatrixDotScalar(i, c->raw_real[r-x+1], c->raw_imag[r-x+1], 1, e, &e[1]);
		}else{
			wekuaMatrixDotScalar(i, c->raw_real[r-x+1], 0.0, 1, e, &e[1]);
		}

		wekuaMatrixAdd(b[1], i, 1, &e[1], &e[2]);
		wekuaFreeMatrix(i, 1, &e[2]);		

		for (uint32_t x=0; x<3; x++) clReleaseEvent(e[x]);

		i = wekuaMatrixProduct(a, b[1], 0, NULL, e);
		if (a->com){
			wekuaMatrixTrace(i, &c->raw_real[a->shape[0]-x], &c->raw_imag[r-x], 1, e);
			c->raw_imag[r-x] /= -1.0*x;
		}else{
			wekuaMatrixTrace(i, &c->raw_real[a->shape[0]-x], NULL, 1, e);
			c->raw_real[r-x] /= -1.0*x;
		}
		wekuaFreeMatrix(i, 0, NULL);
		wekuaFreeMatrix(b[0], 0, NULL);

		b[0] = wekuaMatrixCopy(b[1], 0, NULL, &e[1]);
		wekuaFreeMatrix(b[1], 1, &e[1]);

		for (uint32_t x=0; x<2; x++) clReleaseEvent(e[x]);
	}
	wekuaFreeMatrix(b[0], 0, NULL);
	free(b);
	return c;
}

wmatrix *wekuaMatrixPower(wmatrix *a, int64_t n){
	if (a == NULL){
		return NULL;
	}else if (a->shape[0] != a->shape[1]){
		return NULL;
	}else if (n == 0){
		return wekuaMatrixIden(a->ctx, a->shape[1]);
	}
	wmatrix **b, *c;
	uint8_t d=0;
	b = (wmatrix**) calloc(2, sizeof(wmatrix*));
	cl_event e;

	if (n < 0){
		b[0] = wekuaMatrixInv(a, 0, NULL);
		c = wekuaMatrixPower(b[0], -1*n);
	}else{
		b[0] = a;
		for (int64_t x=1; x<n; x++){
			b[d] = wekuaMatrixProduct(b[d], a, 0, NULL, &e);
			d ^= 1;
			wekuaFreeMatrix(b[d], 1, &e);
			clReleaseEvent(e);
		}
		if (n != 1){
			d ^= 1;
		}
		c = b[d];
	}
	free(b);
	return c;
}
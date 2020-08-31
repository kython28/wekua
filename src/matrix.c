#include "wekua.h"
#include <unistd.h>
#include <math.h>

uint64_t getWI(uint64_t a, uint64_t max, uint32_t cu){
	uint64_t x;
	if (a == 1 || max == 1){
		return 1;
	}else if (a <= cu){
		return a;
	}
	for (x=2; max < a/x || a%x != 0; x++);
	return a/x;
}

void getLWI(uint64_t *x, uint64_t *y, uint32_t si, uint64_t max){
	uint64_t c = 1;
	for (uint32_t j=0; j<si; j++){
		y[j] = x[j];
		c *= x[j];
	}
	if (c <= max){
		return;
	}
	c = (uint64_t) pow(1.0*max, 1.0/si);
	for (uint32_t j=0; j<si; j++){
		if (x[j] < c){
			y[j] = x[j];
			continue;
		}
		y[j] = c;
		while (x[j]%y[j] != 0){
			y[j]--;
		}
	}
}

uint64_t getCol(wmatrix *a){
	if (a->parent != NULL){
		return ((wmatrix*)a->parent)->shape[1];
	}else{
		return a->shape[1];
	}
}

uint64_t getRow(wmatrix *a){
	if (a->parent != NULL){
		return ((wmatrix*)a->parent)->shape[0];
	}else{
		return a->shape[0];
	}
}

void MapBufferMatrix(wmatrix *a){
	if (a->parent != NULL){
		return;
	}

	if (a->real != NULL && a->raw_real == NULL){
		a->raw_real = clEnqueueMapBuffer(a->ctx->command_queue, a->real, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);	
	}
	if (a->imag != NULL && a->raw_imag == NULL){
		a->raw_imag = clEnqueueMapBuffer(a->ctx->command_queue, a->imag, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);
	}
}

void UnmapBufferMatrix(wmatrix *a){
	if (a->parent != NULL){
		return;
	}

	if (a->real != NULL && a->raw_real != NULL){
		clEnqueueUnmapMemObject(a->ctx->command_queue, a->real, a->raw_real, 0, NULL, NULL);
		a->raw_real = NULL;
	}
	if (a->imag != NULL && a->raw_imag != NULL){
		clEnqueueUnmapMemObject(a->ctx->command_queue, a->imag, a->raw_imag, 0, NULL, NULL);
		a->raw_imag = NULL;
	}
}

void runKernel(cl_command_queue cmd, cl_kernel kernel, uint32_t ndim, uint64_t *offsi, uint64_t *glosi, uint64_t *losi){
	cl_event event;
	uint32_t ret = clEnqueueNDRangeKernel(cmd, kernel, ndim, offsi, glosi, losi, 0, NULL, &event);
	if (ret != 0){
		printf("Failed to run kernel %i :-(\n", ret);
		exit(1);
	}
	ret = clWaitForEvents(1, &event);
	if (ret != 0){
		printf("Failed to run kernel %i :-(\n", ret);
		exit(1);
	}
	clReleaseEvent(event);
}

uint8_t createComplexMatrix(wmatrix *b){
	wmatrix *a = NULL;
	if (b == NULL){
		return 1;
	}else if (b->parent != NULL){
		a = b->parent;
	}else{
		a = b;
	}

	if (a->com){
		return 0;
	}
	int ret;
	a->imag = clCreateBuffer(a->ctx->ctx, CL_MEM_READ_WRITE, sizeof(double)*a->size, NULL, &ret);
	if (ret != 0){
		printf("Failed to allocate new memory :-(\n");
		return 1;
	}
	MapBufferMatrix(a);
	cl_event ie;
	double beta = 0.0;
	clEnqueueFillBuffer(a->ctx->command_queue, a->imag, &beta, sizeof(double), 0, a->size*sizeof(double), 0, NULL, &ie);
	clWaitForEvents(1, &ie);
	clReleaseEvent(ie);
	a->com = 1;
	if (b->parent != NULL){
		b->imag = a->imag;
		b->raw_imag = &a->raw_imag[b->offset[0]*a->shape[1]+b->offset[1]];
		b->com = 1;
	}

	return 0;
}

void removeComplexMatrix(wmatrix *b){
	wmatrix *a;
	if (b == NULL){
		return;
	}else if (b->parent != NULL){
		a = b->parent;
	}else{
		a = b;
	}
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->imag, a->raw_imag, 0, NULL, NULL);
	a->raw_imag = NULL;
	clReleaseMemObject(a->imag);
	a->imag = NULL;
	a->com = 0;
	if (a != NULL){
		b->raw_imag = NULL;
	}
}

void wekuaFreeMatrix(wmatrix *a){
	if (a == NULL){
		return;
	}else if (a->parent != NULL){
		free(a);
		return;
	}
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
	uint64_t col = getCol(a);
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
				printf("%14.5e", a->raw_real[y*col+x]);
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
	uint64_t col = getCol(a);
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
				sprintf(num, "%.1e%+.1ej", a->raw_real[y*col+x], a->raw_imag[y*col+x]);
				printf("%21s", num);
				if (y+1 != a->shape[0] || x+1 != a->shape[1]){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= a->shape[0]-2)) || (y == 2 && (x < 3 || x >= a->shape[1]-2))){
				printf("%22s", "... ");
			}
			if (a->shape[0] > 1 && x == a->shape[1]-1 && (y < 3 || y >= a->shape[0]-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixPrint(wmatrix *a){
	if (a == NULL){
		return;
	}
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
	a->parent = NULL;
	a->com = 0;
	a->shape[0] = r;
	a->shape[1] = c;
	a->size = r*c;
	a->ctx = ctx;
	a->work_items[0] = getWI(a->size, max, cu);
	a->work_items[3] = getWI(a->shape[0], max, cu);
	a->work_items[4] = getWI(a->shape[1], max, cu);
	getLWI(a->shape, &a->work_items[1], 2, max);

	int ret;
	a->real = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, sizeof(double)*a->size, NULL, &ret);
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
		wekuaFreeMatrix(a);
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
			wekuaFreeMatrix(a);
			return NULL;
		}
	}
	cl_event e, ie;
	clEnqueueFillBuffer(a->ctx->command_queue, a->real, &alpha, sizeof(double), 0, a->size*sizeof(double), 0, NULL, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);
	if (beta != 0.0){
		clEnqueueFillBuffer(a->ctx->command_queue, a->imag, &beta, sizeof(double), 0, a->size*sizeof(double), 0, NULL, &ie);
		clWaitForEvents(1, &ie);
		clReleaseEvent(ie);
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
			wekuaFreeMatrix(a);
			return NULL;
		}
	}

	cl_kernel kernel = ctx->kernels[0];

	cl_mem ran_r=NULL, ran_i=NULL;
	uint64_t *ran_r_m, *ran_i_m;
	int ret;
	
	ran_r = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, a->size*8, NULL, &ret);
	if (ret != 0){
		clReleaseMemObject(ran_r);
		return NULL;
	}
	ran_r_m = clEnqueueMapBuffer(ctx->command_queue, ran_r, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*8, 0, 0, NULL, NULL);
	getRandomBuffer(ran_r_m, a->size*8);
	clEnqueueUnmapMemObject(ctx->command_queue, ran_r, ran_r_m, 0, NULL, NULL);

	if (a->com){
		ran_i = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, a->size*8, NULL, &ret);
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
	runKernel(a->ctx->command_queue, kernel, 1, NULL, &a->size, a->work_items);

	clReleaseMemObject(ran_r);
	clReleaseMemObject(ran_i);

	return a;
}

wmatrix *wekuaMatrixRandUniform(wekuaContext *ctx, uint64_t r, uint64_t c, double ra, double ia, double re, double ie, uint8_t com){
	if (ctx == NULL){
		return NULL;
	}
	cl_kernel kernel = ctx->kernels[20];
	wmatrix *a = wekuaMatrixRandn(ctx, r, c, com);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(double), &ra);
	clSetKernelArg(kernel, 3, sizeof(double), &ia);
	clSetKernelArg(kernel, 4, sizeof(double), &re);
	clSetKernelArg(kernel, 5, sizeof(double), &ie);
	clSetKernelArg(kernel, 6, 8, &a->shape[1]);
	clSetKernelArg(kernel, 7, 1, &com);

	runKernel(a->ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1]);

	return a;
}

wmatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint64_t r, uint64_t c, void *rbuf, void *ibuf){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	memcpy(a->raw_real, rbuf, a->size*sizeof(double));
	if (ibuf != NULL){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a);
			return NULL;
		}
		memcpy(a->raw_imag, ibuf, a->size*sizeof(double));
	}
	return a;
}

wmatrix *wekuaMatrixIden(wekuaContext *ctx, uint64_t c){
	cl_kernel kernel = ctx->kernels[1];
	wmatrix *i = wekuaFillMatrix(ctx, c, c, 0.0, 0.0);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &i->real);
	clSetKernelArg(kernel, 1, 8, &i->shape[0]);
	runKernel(i->ctx->command_queue, kernel, 1, NULL, &i->shape[1], &i->work_items[4]);
	return i;
}

wmatrix *wekuaMatrixTrans(wmatrix *a){
	if (a == NULL){
		return NULL;
	}
	wekuaContext *ctx = a->ctx;
	wmatrix *b;
	uint64_t col;
	if (a->com){
		b = wekuaAllocComplexMatrix(ctx, a->shape[1], a->shape[0]);
	}else{
		b = wekuaAllocMatrix(ctx, a->shape[1], a->shape[0]);
	}

	cl_kernel kernel = ctx->kernels[2];
	col = getCol(a);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 4, 8, &col);
	clSetKernelArg(kernel, 5, 8, &a->shape);
	clSetKernelArg(kernel, 6, 1, &a->com);
	clSetKernelArg(kernel, 7, 8, a->offset);
	clSetKernelArg(kernel, 8, 8, &a->offset[1]);

	runKernel(a->ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1]);

	return b;
}

wmatrix *wekuaMatrixCopy(wmatrix *a){
	wmatrix *b;
	if (a == NULL){
		return NULL;
	}else if (a->parent != NULL){
		b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
		wekuaMatrixDot(b, a);
		return b;
	}
	wekuaContext *ctx = a->ctx;
	b = (wmatrix*) malloc(sizeof(wmatrix));
	b->parent = NULL;
	b->com = 0;
	memcpy(b->shape, a->shape, 16);
	memcpy(b->offset, a->offset, 16);
	memcpy(b->work_items, a->work_items, 40);
	b->size = a->size;
	b->ctx = ctx;

	int ret;
	b->real = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, sizeof(double)*b->size, NULL, &ret);
	if (ret != 0){
		printf("Failed to allocate new memory :-(\n");
		free(b);
		return NULL;
	}

	b->raw_real = NULL;

	b->imag = NULL;
	b->raw_imag = NULL;

	MapBufferMatrix(b);

	cl_event e, ie;
	clEnqueueCopyBuffer(ctx->command_queue, a->real, b->real, 0, 0, a->size*sizeof(double), 0, NULL, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);
	if (a->com){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(b);
			return NULL;
		}
		clEnqueueCopyBuffer(ctx->command_queue, a->imag, b->imag, 0, 0, a->size*sizeof(double), 0, NULL, &ie);
		clWaitForEvents(1, &ie);
		clReleaseEvent(ie);
	}
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
	b->parent = a;
	b->ctx = a->ctx;
	b->size = w*h;
	b->com = a->com;

	b->shape[0] = h;
	b->shape[1] = w;
	b->offset[0] = y;
	b->offset[1] = x;

	b->real = a->real;
	b->imag = a->imag;

	b->raw_real = &a->raw_real[y*a->shape[1]+x];
	b->raw_imag = NULL;
	if (a->com){
		b->raw_imag = &a->raw_imag[y*a->shape[1]+x];
	}

	b->work_items[0] = getWI(b->size, max, cu);
	b->work_items[3] = getWI(b->shape[0], max, cu);
	b->work_items[4] = getWI(b->shape[1], max, cu);
	getLWI(b->shape, &b->work_items[1], 2, max);

	return b;
}

wmatrix *wekuaMatrixResize(wmatrix *a, uint64_t r, uint64_t c, double alpha, double beta){
	if (a == NULL){
		return NULL;
	}else if (a->parent != NULL){
		return NULL;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[16];

	wmatrix *b = wekuaFillMatrix(a->ctx, r, c, alpha, beta);
	if (a->com || beta != 0.0){
		if (createComplexMatrix(b) || createComplexMatrix(a)){
			wekuaFreeMatrix(b);
			return NULL;
		}
	}

	uint64_t shape[2], wi[2];
	shape[0] = b->shape[0];
	shape[1] = b->shape[1];

	if (shape[0] > a->shape[0]){
		shape[0] = a->shape[0];
	}

	if (shape[1] > a->shape[1]){
		shape[1] = a->shape[1];
	}

	getLWI(shape, wi, 2, ctx->max_work_group_size);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 4, 8, &b->shape[1]);
	clSetKernelArg(kernel, 5, 8, &a->shape[1]);
	clSetKernelArg(kernel, 6, 1, &b->com);

	runKernel(ctx->command_queue, kernel, 2, NULL, shape, wi);

	return b;
}

wmatrix *wekuaMatrixProduct(wmatrix *a, wmatrix *b){
	if (a == NULL || b == NULL){
		return NULL;
	}else if (a->shape[1] != b->shape[0]){
		return NULL;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[4];

	uint64_t col, col2;
	wmatrix *c = wekuaAllocMatrix(ctx, a->shape[0], b->shape[1]);
	if (c == NULL){
		return NULL;
	}
	if (a->com || b->com){
		if (createComplexMatrix(a) || createComplexMatrix(b) || createComplexMatrix(c)){
			wekuaFreeMatrix(c);
			return NULL;
		}
	}
	col = getCol(a); col2 = getCol(b);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &c->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &c->imag);
	clSetKernelArg(kernel, 6, sizeof(uint8_t), &a->com);
	clSetKernelArg(kernel, 7, 8, &col);
	clSetKernelArg(kernel, 8, 8, &col2);
	clSetKernelArg(kernel, 9, 8, a->offset);
	clSetKernelArg(kernel, 10, 8, &a->offset[1]);
	clSetKernelArg(kernel, 11, 8, b->offset);
	clSetKernelArg(kernel, 12, 8, &b->offset[1]);

	runKernel(ctx->command_queue, kernel, 2, NULL, c->shape, &c->work_items[1]);
	return c;
}

void Axpy(wmatrix *a, wmatrix *b, double alpha){
	if (a == NULL || b == NULL){
		return;
	}else if(memcmp(a->shape, b->shape, 16) != 0){
		return;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[3];

	uint64_t col, col2;

	if (a->com && b->com == 0){
		if (createComplexMatrix(b)){
			return;
		}
	}else if (b->com && a->com == 0){
		if (createComplexMatrix(a)){
			return;
		}
	}

	col = getCol(a); col2 = getCol(b);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 1, &a->com);
	clSetKernelArg(kernel, 5, sizeof(double), &alpha);
	clSetKernelArg(kernel, 6, 8, &col);
	clSetKernelArg(kernel, 7, 8, &col2);
	clSetKernelArg(kernel, 8, 8, a->offset);
	clSetKernelArg(kernel, 9, 8, &a->offset[1]);
	clSetKernelArg(kernel, 10, 8, b->offset);
	clSetKernelArg(kernel, 11, 8, &b->offset[1]);

	runKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1]);
}

void wekuaMatrixAdd(wmatrix *a, wmatrix *b){
	Axpy(a, b, 1.0);
}

void wekuaMatrixSub(wmatrix *a, wmatrix *b){
	Axpy(a, b, -1.0);
}

void wekuaMatrixDotScalar(wmatrix *a, double alpha, double beta){
	if (a == NULL){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[11];

	uint64_t col = getCol(a);
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
	clSetKernelArg(kernel, 5, 8, &col);

	runKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1]);
}

void wekuaMatrixDot(wmatrix *a, wmatrix *b){
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
	cl_kernel kernel = ctx->kernels[25];

	uint64_t col = getCol(a), col2 = getCol(b);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 1, &a->com);
	clSetKernelArg(kernel, 5, 8, &col);
	clSetKernelArg(kernel, 6, 8, &col2);
	clSetKernelArg(kernel, 7, 8, a->offset);
	clSetKernelArg(kernel, 8, 8, &a->offset[1]);
	clSetKernelArg(kernel, 9, 8, b->offset);
	clSetKernelArg(kernel, 10, 8, &b->offset[1]);

	runKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1]);
}

void wekuaMatrixAbs(wmatrix *a){
	if (a == NULL){
		return;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[12];

	uint64_t col = getCol(a);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 1, &a->com);
	clSetKernelArg(kernel, 3, 8, &col);

	runKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1]);

	removeComplexMatrix(a);
}

void wekuaMatrixAbsdiff(wmatrix *a, wmatrix *b){
	if (a == NULL || b == NULL){
		return;
	}
	wekuaMatrixSub(a, b);
	wekuaMatrixAbs(a);
}

void wekuaMatrixLn(wmatrix *a){
	if (a == NULL){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[24];

	uint64_t col = getCol(a);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 1, &a->com);
	clSetKernelArg(kernel, 3, 8, &col);

	runKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1]);
}

void wekuaMatrixLog(wmatrix *a, double r_base, double i_base){
	if (a == NULL){
		return;
	}
	wmatrix *b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], r_base, i_base);
	wekuaMatrixLn(a);
	wekuaMatrixLn(b);
	wekuaMatrixDivide(a, b);
	wekuaFreeMatrix(b);
}

void wekuaMatrixDivide(wmatrix *a, wmatrix *b){
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
	cl_kernel kernel = ctx->kernels[26];
	uint64_t col = getCol(a), col2 = getCol(b);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, 1, &a->com);
	clSetKernelArg(kernel, 5, 8, &col);
	clSetKernelArg(kernel, 6, 8, &col2);
	clSetKernelArg(kernel, 7, 8, a->offset);
	clSetKernelArg(kernel, 8, 8, &a->offset[1]);
	clSetKernelArg(kernel, 9, 8, b->offset);
	clSetKernelArg(kernel, 10, 8, &b->offset[1]);

	runKernel(ctx->command_queue, kernel, 2, NULL, a->shape, &a->work_items[1]);
}

void wekuaMatrixPowr(wmatrix *a, double real, double imag){
	if (a == NULL){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[27];

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

	runKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1]);
}

wmatrix *wekuaMatrixDiag(wmatrix *a){
	wmatrix *b, *c;
	if (a == NULL){
		return NULL;
	}else if (a->parent != NULL){
		return NULL;
	}else if (a->shape[1] == a->shape[0]){
		b = wekuaFillMatrix(a->ctx, 1, a->shape[0], 1.0, 0.0);
		c = wekuaMatrixProduct(a, b);
		wekuaFreeMatrix(b);
		return c;
	}else if (a->shape[1] > 1 && a->shape[0] > 1){
		return NULL;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[13];

	uint64_t dim = a->shape[0];
	if (dim < a->shape[1]){
		dim = a->shape[1];
	}
	b = wekuaFillMatrix(a->ctx, dim, dim, 0.0, 0.0);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
			return NULL;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 4, 1, &a->com);

	runKernel(ctx->command_queue, kernel, 1, NULL, &a->size, a->work_items);
	return b;
}

wmatrix *wekuaArange(wekuaContext *ctx, double x, double y, double alpha){
	wmatrix *a;
	int64_t col = fabs((y-x)/alpha);
	while (x+col*alpha > y && col > 0){
		col--;
	}

	cl_kernel kernel = ctx->kernels[28];
	a = wekuaFillMatrix(ctx, 1, (uint32_t)col, x, 0.0);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(double), &alpha);

	runKernel(ctx->command_queue, kernel, 1, NULL, &a->size, a->work_items);

	return a;
}

void wekuaMatrixSum(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}

	wmatrix *b, *c, *d;
	b = wekuaFillMatrix(a->ctx, a->shape[1], 1, 1.0, 0.0);
	d = wekuaFillMatrix(a->ctx, a->shape[0], 1, 1.0, 0.0);
	
	c = wekuaMatrixProduct(a, b);
	c->shape[1] = c->shape[0];
	c->shape[0] = 1;

	d = wekuaMatrixProduct(c, d);


	if (real != NULL){
		real[0] = d->raw_real[0];
	}
	if (a->com && imag != NULL){
		imag[0] = d->raw_imag[0];
	}

	wekuaFreeMatrix(b);
	wekuaFreeMatrix(c);
	wekuaFreeMatrix(d);
}

void wekuaMatrixMul(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[14];

	double re=1.0, im=1.0;
	uint64_t col = getCol(a);

	wmatrix *b;

	if (a->shape[1] == 1 || a->shape[0] == 1){
		b = wekuaMatrixCopy(a);
		goto Sumstetwo;
	}

	b = wekuaFillMatrix(ctx, 1, a->shape[0], 1.0, 0.0);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
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

	runKernel(ctx->command_queue, kernel, 1, NULL, a->shape, &a->work_items[3]);

	Sumstetwo:

	if (a->com){
		for (uint64_t i=0; i<b->size; i++){
			re = b->raw_real[i]*re - b->raw_imag[i]*im;
			im = b->raw_real[i]*im + b->raw_imag[i]*re;
		}
	}else{
		for (uint64_t i=0; i<b->size; i++){
			re *= b->raw_real[i];
		}
	}

	if (real != NULL){
		real[0] = re;
	}
	if (a->com && imag != NULL){
		imag[0] = im;
	}

	wekuaFreeMatrix(b);
}

void wekuaMatrixMean(wmatrix *a, double *real, double *imag){
	wmatrix *b = wekuaMatrixCopy(a);
	double n = 1.0/b->size;
	wekuaMatrixDotScalar(b, n, 0.0);
	wekuaMatrixSum(b, real, imag);
	wekuaFreeMatrix(b);
}

void wekuaMatrixNorm(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[15];

	wmatrix *b = wekuaMatrixCopy(a);

	uint64_t col = getCol(a);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 2, 1, &b->com);
	clSetKernelArg(kernel, 3, 8, &col);

	runKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1]);

	wekuaMatrixSum(b, real, imag);

	double ang, n;

	if (a->com){
		if (real[0] == 0){
			ang = CL_M_PI_2;
		}else{
			ang = atan(imag[0]/real[0])/2;
		}
		n = sqrt(real[0]*real[0]+imag[0]*imag[0]);
		real[0] = n*cos(ang);
		imag[0] = n*sin(ang);
	}else{
		real[0] = sqrt(real[0]);
	}
	wekuaFreeMatrix(b);
}

void wekuaMatrixDet(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}else if (a->shape[0] != a->shape[1]){
		return;
	}else if (a->parent != NULL){
		return;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[17];
	cl_command_queue cmd = ctx->command_queue;

	wmatrix *b = wekuaMatrixCopy(a);
	wmatrix *c = wekuaFillMatrix(ctx, a->shape[0], a->shape[1], 1.0, 0.0);
	if (b->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(c);
			wekuaFreeMatrix(b);
			return;
		}
	}

	cl_event *event = (cl_event*) malloc(sizeof(cl_event)*(a->shape[1]-1));
	cl_event *befo = NULL;
	uint64_t we=0;

	for (uint64_t k=0; k < a->shape[1]-1; k++){
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &c->real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &c->imag);
		clSetKernelArg(kernel, 4, 8, &k);
		clSetKernelArg(kernel, 5, 8, &b->shape[1]);
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

	wekuaMatrixMul(c, real, imag);

	wekuaFreeMatrix(b);
	wekuaFreeMatrix(c);
}

wmatrix *wekuaMatrixInv(wmatrix *a){
	if (a == NULL){
		return NULL;
	}else if (a->shape[0] != a->shape[1]){
		return NULL;
	}else if (a->parent != NULL){
		return NULL;
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[18], kernel2 = ctx->kernels[19];
	cl_command_queue cmd = ctx->command_queue;

	wmatrix *inv, *b;
	inv = wekuaMatrixIden(a->ctx, a->shape[0]);
	if (a->com){
		if (createComplexMatrix(inv)){
			wekuaFreeMatrix(inv);
			return NULL;
		}
	}

	b = wekuaMatrixCopy(a);

	cl_event *event = (cl_event*) malloc(sizeof(cl_event)*(2*(a->shape[1]-1)+1));
	cl_event *befo = NULL;
	uint32_t we=0;

	uint8_t otherm=1;

	for (uint8_t t=0; t<2; t++){
		for (uint64_t k=0; k < a->shape[1]-1; k++){
			clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &inv->real);
			clSetKernelArg(kernel, 3, sizeof(cl_mem), &inv->imag);
			clSetKernelArg(kernel, 4, 8, &k);
			clSetKernelArg(kernel, 5, 8, &a->shape[1]);
			clSetKernelArg(kernel, 6, 1, &a->com);
			clSetKernelArg(kernel, 7, 1, &otherm);
			clSetKernelArg(kernel, 8, 1, &t);
			clSetKernelArg(kernel, 9, 1, &otherm);

			clEnqueueNDRangeKernel(cmd, kernel, 1, NULL, a->shape, &a->work_items[3], we, befo, &event[t*(a->shape[1]-1)+k]);
			if (we == 0){
				we++;
			}
			befo = &event[t*(a->shape[1]-1)+k];
		}
	}
	clSetKernelArg(kernel2, 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel2, 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel2, 2, sizeof(cl_mem), &inv->real);
	clSetKernelArg(kernel2, 3, sizeof(cl_mem), &inv->imag);
	clSetKernelArg(kernel2, 4, 8, &a->shape[1]);
	clSetKernelArg(kernel2, 5, 1, &a->com);

	clEnqueueNDRangeKernel(cmd, kernel2, 2, NULL, a->shape, &a->work_items[1], 1, befo, &event[2*(a->shape[1]-1)]);

	clWaitForEvents(1, &event[2*(a->shape[1]-1)]);
	for (uint64_t x=0; x < 2*(a->shape[1]-1)+1; x++){
		clReleaseEvent(event[x]);
	}
	free(event);
	wekuaFreeMatrix(b);
	return inv;
}

wmatrix *wekuaMatrixSolve(wmatrix *a, wmatrix *b){
	wmatrix *c = wekuaMatrixInv(a);
	wmatrix *d = wekuaMatrixProduct(c, b);
	wekuaFreeMatrix(c);
	return d;
}

uint32_t wekuaMatrixRang(wmatrix *a){
	if (a == NULL){
		return 0;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[18];
	cl_command_queue cmd = ctx->command_queue;

	uint32_t rang=0;
	wmatrix *b, *c, *e;
	e = wekuaFillMatrix(a->ctx, a->shape[1], 1, 0.0, 0.0);
	if (a->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(c);
			return rang;
		}
	}
	b = wekuaMatrixCopy(a);
	cl_mem nullptr = NULL;
	uint64_t wi[2];
	wi[0] = a->work_items[3];
	wi[1] = 1;

	uint8_t d = 0;
	if (a->shape[1] > a->shape[0]){
		d = 1;
	}

	cl_event *event = (cl_event*) malloc(sizeof(cl_event)*(a->shape[1]-1));
	cl_event *befo = NULL;
	uint32_t we=0;

	uint8_t otherm=0, t=0;

	for (uint64_t k=0; k < a->shape[1]-1; k++){
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &nullptr);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &nullptr);
		clSetKernelArg(kernel, 4, 8, &k);
		clSetKernelArg(kernel, 5, 8, &a->shape[1]);
		clSetKernelArg(kernel, 6, 1, &a->com);
		clSetKernelArg(kernel, 7, 1, &otherm);
		clSetKernelArg(kernel, 8, 1, &t);
		clSetKernelArg(kernel, 9, 1, &otherm);

		clEnqueueNDRangeKernel(cmd, kernel, 1, NULL, &a->shape[d], &a->work_items[3+d], we, befo, &event[k]);
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

	wekuaMatrixAbs(b);

	c = wekuaMatrixProduct(a, e);
	wekuaFreeMatrix(e);

	for (uint64_t r=0; r < c->size; r++){
		if (c->raw_real[r] > CL_DBL_EPSILON){
			rang++;
		}
	}
	wekuaFreeMatrix(b);
	wekuaFreeMatrix(c);
	
	return rang;
}

wmatrix *wekuaMatrixPinv(wmatrix *a){
	if (a == NULL){
		return NULL;
	}
	wmatrix *pinv=NULL, *ta=NULL, *te=NULL, *ti=NULL;
	ta = wekuaMatrixTrans(a);	
	uint32_t rang = wekuaMatrixRang(a);
	if (rang == a->shape[0]){
		te = wekuaMatrixProduct(a, ta);
		ti = wekuaMatrixInv(te);
		pinv = wekuaMatrixProduct(ta, ti);
	}else if (rang == a->shape[1]){
		te = wekuaMatrixProduct(ta, a);
		ti = wekuaMatrixInv(te);
		pinv = wekuaMatrixProduct(ti, ta);
	}
	wekuaFreeMatrix(ta);
	wekuaFreeMatrix(te);
	wekuaFreeMatrix(ti);
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

void wekuaMatrixMax(wmatrix *a, double *real, double *imag){
	uint64_t i, col = getCol(a);
	if (a == NULL){
		return;
	}else if (a->com && real != NULL && imag != NULL){
		real[0] = a->raw_real[0]; imag[0] = a->raw_imag[0];
		for (uint64_t y=0; y<a->shape[0]; y++){
			for (uint64_t x=1; x < a->shape[1]; x++){
				i = y*col+x;
				if (sqrt(a->raw_real[i]*a->raw_real[i] + a->raw_imag[i]*a->raw_imag[i]) > sqrt(real[0]*real[0] + imag[0]*imag[0])){
					real[0] = a->raw_real[i];
					imag[0] = a->raw_imag[i];
				}
			}
		}
	}else if (a->com == 0 && real != NULL){
		real[0] = a->raw_real[0];
		for (uint64_t y=0; y<a->shape[0]; y++){
			for (uint64_t x=1; x < a->shape[1]; x++){
				if (a->raw_real[y*col+x] > real[0]){
					real[0] = a->raw_real[y*col+x];
				}
			}
		}
	}
}

void wekuaMatrixMin(wmatrix *a, double *real, double *imag){
	uint64_t i, col = getCol(a);
	if (a == NULL){
		return;
	}else if (a->com && real != NULL && imag != NULL){
		real[0] = a->raw_real[0]; imag[0] = a->raw_imag[0];
		for (uint64_t y=0; y<a->shape[0]; y++){
			for (uint64_t x=1; x < a->shape[1]; x++){
				i = y*col+x;
				if (sqrt(a->raw_real[i]*a->raw_real[i] + a->raw_imag[i]*a->raw_imag[i]) < sqrt(real[0]*real[0] + imag[0]*imag[0])){
					real[0] = a->raw_real[i];
					imag[0] = a->raw_imag[i];
				}
			}
		}
	}else if (a->com == 0 && real != NULL){
		real[0] = a->raw_real[0];
		for (uint64_t y=0; y<a->shape[0]; y++){
			for (uint64_t x=1; x < a->shape[1]; x++){
				if (a->raw_real[y*col+x] < real[0]){
					real[0] = a->raw_real[y*col+x];
				}
			}
		}
	}
}

void wekuaMatrixToComplex(wmatrix *a, double *real, double *imag){
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
	wekuaFreeMatrix(a);
	return b;
}

void wekuaMatrixTrace(wmatrix *a, double *real, double *imag){
	wmatrix *b = wekuaMatrixDiag(a);
	wekuaMatrixSum(b, real, imag);
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
			wekuaFreeMatrix(c);
			return NULL;
		}
	}
	b = (wmatrix**) malloc(sizeof(wmatrix*)*2);
	b[0] = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 0.0, 0.0);
	c->raw_real[a->shape[0]] = 1.0;
	for (uint64_t x=1; x<=a->shape[0]; x++){
		i = wekuaMatrixIden(a->ctx, a->shape[0]);
		b[1] = wekuaMatrixProduct(a, b[0]);
		if (a->com){
			wekuaMatrixDotScalar(i, c->raw_real[a->shape[0]-x+1], c->raw_imag[a->shape[0]-x+1]);
		}else{
			wekuaMatrixDotScalar(i, c->raw_real[a->shape[0]-x+1], 0.0);
		}
		wekuaMatrixAdd(b[1], i);
		wekuaFreeMatrix(i);
		i = wekuaMatrixProduct(a, b[1]);
		if (a->com){
			wekuaMatrixTrace(i, &c->raw_real[a->shape[0]-x], &c->raw_imag[a->shape[0]-x]);
			c->raw_imag[a->shape[0]-x] /= -1.0*x;
		}else{
			wekuaMatrixTrace(i, &c->raw_real[a->shape[0]-x], NULL);
			c->raw_real[a->shape[0]-x] /= -1.0*x;
		}
		wekuaFreeMatrix(i);
		wekuaFreeMatrix(b[0]);
		b[0] = wekuaMatrixCopy(b[1]);
		wekuaFreeMatrix(b[1]);
	}
	wekuaFreeMatrix(b[0]);
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
	if (n < 0){
		b[0] = wekuaMatrixInv(a);
		c = wekuaMatrixPower(b[0], -1*n);
	}else{
		b[0] = a;
		for (int64_t x=1; x<n; x++){
			b[d] = wekuaMatrixProduct(b[d], a);
			d ^= 1;
			wekuaFreeMatrix(b[d]);
		}
		if (n != 1){
			d ^= 1;
		}
		c = b[d];
	}
	free(b);
	return c;
}
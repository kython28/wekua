#include "wekua.h"
#include <unistd.h>
#include <math.h>

uint64_t getWI(uint64_t a, uint64_t max){
	if (a == 1 || max == 1){
		return 1;
	}else if (a <= max){
		return a;
	}
	uint64_t x;
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

void MapBufferMatrix(wmatrix *a){
	if (a->real != NULL && a->raw_real == NULL){
		a->raw_real = clEnqueueMapBuffer(a->ctx->command_queue, a->real, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);	
	}
	if (a->imag != NULL && a->raw_imag == NULL){
		a->raw_imag = clEnqueueMapBuffer(a->ctx->command_queue, a->imag, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);
	}
}

void UnmapBufferMatrix(wmatrix *a){
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
	}
	ret = clWaitForEvents(1, &event);
	if (ret != 0){
		printf("Failed to run kernel %i :-(\n", ret);
	}
	clReleaseEvent(event);
}

uint8_t createComplexMatrix(wmatrix *a){
	if (a->com){
		return 0;
	}
	int ret;
	a->imag = clCreateBuffer(a->ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(double)*a->size, NULL, &ret);
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
	return 0;
}

void removeComplexMatrix(wmatrix *a){
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->imag, a->raw_imag, 0, NULL, NULL);
	a->raw_imag = NULL;
	clReleaseMemObject(a->imag);
	a->imag = NULL;
	a->com = 0;
}

void wekuaFreeMatrix(wmatrix *a){
	if (a == NULL){
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
				printf("%14.5e", a->raw_real[y*a->shape[1]+x]);
				if (y+1 != a->shape[0] || x+1 != a->shape[1]){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= a->shape[0]-4)) || (y == 4 && (x < 4 || x >= a->shape[1]-4))){
				printf("%15s", "... ");
			}
			if (x == a->shape[1]-1 && (y < 5 || y >= a->shape[0]-4)){
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
	char num[23];
	for (uint32_t y=0; y<a->shape[0]; y++){
		for (uint32_t x=0; x<a->shape[1]; x++){
			if (x == 0 && (y < 3 || y >= a->shape[0]-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= a->shape[0]-2) && (y < 2 || y >= a->shape[0]-2)){
				memset(num, 0, 23);
				sprintf(num, "%.2e%+.2ei,", a->raw_real[y*a->shape[1]+x], a->raw_imag[y*a->shape[1]+x]);
				printf("%23s,", num);
				if (y+1 != a->shape[0] || x+1 != a->shape[1]){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= a->shape[0]-2)) || (y == 2 && (x < 2 || x >= a->shape[1]-2))){
				printf("%24s", "... ");
			}
			if (x == a->shape[1]-1 && (y < 3 || y >= a->shape[0]-2)){
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
	a->com = 0;
	a->shape[0] = r;
	a->shape[1] = c;
	a->size = r*c;
	a->ctx = ctx;
	a->work_items[0] = getWI(a->size, max);
	a->work_items[3] = getWI(a->shape[0], max);
	a->work_items[4] = getWI(a->shape[1], max);
	getLWI(a->shape, &a->work_items[1], 2, max);

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

	cl_mem ran_r=NULL, ran_i=NULL;
	uint64_t *ran_r_m, *ran_i_m;
	int ret;
	
	ran_r = clCreateBuffer(a->ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, a->size*8, NULL, &ret);
	if (ret != 0){
		clReleaseMemObject(ran_r);
		return NULL;
	}
	ran_r_m = clEnqueueMapBuffer(a->ctx->command_queue, ran_r, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*8, 0, 0, NULL, NULL);
	getRandomBuffer(ran_r_m, a->size*8);
	clEnqueueUnmapMemObject(a->ctx->command_queue, ran_r, ran_r_m, 0, NULL, NULL);

	if (a->com){
		ran_i = clCreateBuffer(a->ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, a->size*8, NULL, &ret);
		if (ret != 0){
			clReleaseMemObject(ran_r);
			clReleaseMemObject(ran_i);
			return NULL;
		}
		ran_i_m = clEnqueueMapBuffer(a->ctx->command_queue, ran_i, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*8, 0, 0, NULL, NULL);
		getRandomBuffer(ran_i_m, a->size*8);
		clEnqueueUnmapMemObject(a->ctx->command_queue, ran_i, ran_i_m, 0, NULL, NULL);
	}

	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[0], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[0], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[0], 2, sizeof(cl_mem), &ran_r);
	clSetKernelArg(a->ctx->kernels[0], 3, sizeof(cl_mem), &ran_i);
	clSetKernelArg(a->ctx->kernels[0], 4, 1, &a->com);
	runKernel(a->ctx->command_queue, a->ctx->kernels[0], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(a);

	clReleaseMemObject(ran_r);
	clReleaseMemObject(ran_i);

	return a;
}

wmatrix *wekuaMatrixRandUniform(wekuaContext *ctx, uint64_t r, uint64_t c, double ra, double ia, double re, double ie, uint8_t com){
	if (ctx == NULL){
		return NULL;
	}
	wmatrix *a = wekuaMatrixRandn(ctx, r, c, com);

	clSetKernelArg(a->ctx->kernels[22], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[22], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[22], 2, sizeof(double), &ra);
	clSetKernelArg(a->ctx->kernels[22], 3, sizeof(double), &ia);
	clSetKernelArg(a->ctx->kernels[22], 4, sizeof(double), &re);
	clSetKernelArg(a->ctx->kernels[22], 5, sizeof(double), &ie);
	clSetKernelArg(a->ctx->kernels[22], 6, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[22], 7, 1, &com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[22], 2, NULL, a->shape, &a->work_items[1]);

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
	wmatrix *i = wekuaFillMatrix(ctx, c, c, 0.0, 0.0);
	clSetKernelArg(i->ctx->kernels[1], 0, sizeof(cl_mem), &i->real);
	clSetKernelArg(i->ctx->kernels[1], 1, 8, &i->shape[0]);
	runKernel(i->ctx->command_queue, i->ctx->kernels[1], 1, NULL, &i->shape[1], &i->work_items[4]);
	return i;
}

wmatrix *wekuaMatrixTrans(wmatrix *a){
	wmatrix *b;
	if (a->com){
		b = wekuaAllocComplexMatrix(a->ctx, a->shape[1], a->shape[0]);
	}else{
		b = wekuaAllocMatrix(a->ctx, a->shape[1], a->shape[0]);
	}

	clSetKernelArg(a->ctx->kernels[2], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[2], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[2], 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[2], 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[2], 4, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[2], 5, 8, &a->shape);
	clSetKernelArg(a->ctx->kernels[2], 6, 1, &a->com);
	runKernel(a->ctx->command_queue, a->ctx->kernels[2], 2, NULL, a->shape, &a->work_items[1]);

	return b;
}

wmatrix *wekuaMatrixCopy(wmatrix *a){
	if (a == NULL){
		return NULL;
	}
	//wmatrix *b = wekuaAllocMatrix(a->ctx, a->shape[0], a->shape[1]);
	wmatrix *b = (wmatrix*) malloc(sizeof(wmatrix));
	b->com = 0;
	memcpy(b->shape, a->shape, 16);
	memcpy(b->work_items, a->work_items, 40);
	b->size = a->size;
	b->ctx = a->ctx;

	int ret;
	b->real = clCreateBuffer(b->ctx->ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(double)*b->size, NULL, &ret);
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
	clEnqueueCopyBuffer(a->ctx->command_queue, a->real, b->real, 0, 0, a->size*sizeof(double), 0, NULL, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);
	if (a->com){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(b);
			return NULL;
		}
		clEnqueueCopyBuffer(a->ctx->command_queue, a->imag, b->imag, 0, 0, a->size*sizeof(double), 0, NULL, &ie);
		clWaitForEvents(1, &ie);
		clReleaseEvent(ie);
	}
	return b;
}

wmatrix *wekuaCutMatrix(wmatrix *a, uint64_t x, uint64_t w, uint64_t y, uint64_t h){
	wmatrix *b = wekuaAllocMatrix(a->ctx, h, w);
	if (w == 0 || h == 0 || x+w+1 > a->shape[1] || y+h+1 > a->shape[0] || w > a->shape[1] || h > a->shape[0] || b == NULL){
		return NULL;
	}

	clSetKernelArg(a->ctx->kernels[3], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[3], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[3], 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[3], 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[3], 4, 8, &b->shape[1]);
	clSetKernelArg(a->ctx->kernels[3], 5, 8, &x);
	clSetKernelArg(a->ctx->kernels[3], 6, 8, &y);
	clSetKernelArg(a->ctx->kernels[3], 7, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[3], 8, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[3], 2, NULL, b->shape, &b->work_items[1]);

	return b;
}

wmatrix *wekuaMatrixResize(wmatrix *a, uint64_t r, uint64_t c, double alpha, double beta){
	if (a == NULL){
		return NULL;
	}
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

	getLWI(shape, wi, 2, a->ctx->max_work_group_size);

	clSetKernelArg(a->ctx->kernels[18], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[18], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[18], 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[18], 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[18], 4, 8, &b->shape[1]);
	clSetKernelArg(a->ctx->kernels[18], 5, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[18], 6, 1, &b->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[18], 2, NULL, shape, wi);

	return b;
}

wmatrix *wekuaMatrixProduct(wmatrix *a, wmatrix *b){
	if (a == NULL || b == NULL){
		return NULL;
	}else if (a->shape[1] != b->shape[0]){
		return NULL;
	}

	wmatrix *c = wekuaAllocMatrix(a->ctx, a->shape[0], b->shape[1]);
	if (a->com || b->com){
		if (createComplexMatrix(a) || createComplexMatrix(b) || createComplexMatrix(c)){
			wekuaFreeMatrix(c);
			return NULL;
		}
	}

	clSetKernelArg(a->ctx->kernels[5], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[5], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[5], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[5], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[5], 4, sizeof(cl_mem), &c->real);
	clSetKernelArg(a->ctx->kernels[5], 5, sizeof(cl_mem), &c->imag);
	clSetKernelArg(a->ctx->kernels[5], 6, sizeof(uint8_t), &a->com);
	clSetKernelArg(a->ctx->kernels[5], 7, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[5], 8, 8, &b->shape[1]);

	runKernel(a->ctx->command_queue, a->ctx->kernels[5], 2, NULL, c->shape, &c->work_items[1]);

	return c;
}

void Axpy(wmatrix *a, wmatrix *b, double alpha){
	if (a == NULL || b == NULL){
		return;
	}else if(memcmp(a->shape, b->shape, 16) != 0){
		return;
	}

	if (a->com && b->com == 0){
		if (createComplexMatrix(b)){
			return;
		}
	}else if (b->com && a->com == 0){
		if (createComplexMatrix(a)){
			return;
		}
	}

	clSetKernelArg(a->ctx->kernels[4], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[4], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[4], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[4], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[4], 4, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[4], 5, 1, &a->com);
	clSetKernelArg(a->ctx->kernels[4], 6, sizeof(double), &alpha);

	runKernel(a->ctx->command_queue, a->ctx->kernels[4], 2, NULL, a->shape, &a->work_items[1]);
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

	if (beta != 0.0 && a->com == 0){
		if (createComplexMatrix(a)){
			return;
		}
	}

	clSetKernelArg(a->ctx->kernels[12], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[12], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[12], 2, 1, &a->com);
	clSetKernelArg(a->ctx->kernels[12], 3, sizeof(double), &alpha);
	clSetKernelArg(a->ctx->kernels[12], 4, sizeof(double), &beta);

	runKernel(a->ctx->command_queue, a->ctx->kernels[12], 1, NULL, &a->size, a->work_items);
}

void wekuaMatrixDot(wmatrix *a, wmatrix *b){
	if (a == NULL || b == NULL){
		return;
	}else if (a->com || b->com){
		if (createComplexMatrix(a) || createComplexMatrix(b)){
			return;
		}
	}

	clSetKernelArg(a->ctx->kernels[29], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[29], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[29], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[29], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[29], 4, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[29], 1, NULL, &a->size, a->work_items);
}

void wekuaMatrixAbs(wmatrix *a){
	if (a == NULL){
		return;
	}

	clSetKernelArg(a->ctx->kernels[13], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[13], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[13], 2, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[13], 2, NULL, &a->size, a->work_items);

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

	clSetKernelArg(a->ctx->kernels[28], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[28], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[28], 2, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[28], 1, NULL, &a->size, a->work_items);
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

	clSetKernelArg(a->ctx->kernels[30], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[30], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[30], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[30], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[30], 4, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[30], 1, NULL, &a->size, a->work_items);
}

void wekuaMatrixPowr(wmatrix *a, double real, double imag){
	if (a == NULL){
		return;
	}

	if (a->com == 0 && imag != 0.0){
		if (createComplexMatrix(a)){
			return;
		}
	}

	clSetKernelArg(a->ctx->kernels[31], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[31], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[31], 2, sizeof(double), &real);
	clSetKernelArg(a->ctx->kernels[31], 3, sizeof(double), &imag);
	clSetKernelArg(a->ctx->kernels[31], 4, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[31], 1, NULL, &a->size, a->work_items);
}

wmatrix *wekuaMatrixDiag(wmatrix *a){
	if (a == NULL){
		return NULL;
	}else if (a->shape[1] != a->shape[0]){
		return NULL;
	}
	wmatrix *b = wekuaAllocMatrix(a->ctx, 1, a->shape[1]);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
			return NULL;
		}
	}

	clSetKernelArg(a->ctx->kernels[14], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[14], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[14], 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[14], 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[14], 4, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[14], 5, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[14], 1, NULL, &b->size, b->work_items);
	return b;
}

wmatrix *wekuaArange(wekuaContext *ctx, double x, double y, double alpha){
	wmatrix *a;
	int64_t col = fabs((y-x)/alpha);
	while (x+col*alpha > y && col > 0){
		col--;
	}

	a = wekuaFillMatrix(ctx, 1, (uint32_t)col, x, 0.0);

	clSetKernelArg(a->ctx->kernels[32], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[32], 1, sizeof(double), &alpha);

	runKernel(a->ctx->command_queue, a->ctx->kernels[32], 1, NULL, &a->size, a->work_items);

	return a;
}

void wekuaMatrixSum(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}

	double re=0.0, im=0.0;
	uint64_t wi[2];

	wmatrix *b;

	if (a->shape[1] == 1 || a->shape[0] == 1){
		b = wekuaMatrixCopy(a);
		goto Sumstetwo;
	}

	b = wekuaFillMatrix(a->ctx, 1, a->shape[0], 0.0, 0.0);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
			return;
		}
	}

	wi[0] = b->work_items[0];
	wi[1] = 1;

	clSetKernelArg(a->ctx->kernels[15], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[15], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[15], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[15], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[15], 4, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[15], 5, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[15], 2, NULL, a->shape, wi);

	Sumstetwo:
	for (uint64_t i=0; i<b->size; i++){
		re += b->raw_real[i];
		if (a->com){
			im += b->raw_imag[i];
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

void wekuaMatrixMul(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}

	double re=1.0, im=1.0;
	uint64_t wi;

	wmatrix *b;

	if (a->shape[1] == 1 || a->shape[0] == 1){
		b = wekuaMatrixCopy(a);
		goto Sumstetwo;
	}

	b = wekuaFillMatrix(a->ctx, 1, a->shape[0], 1.0, 0.0);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
			return;
		}
	}

	wi = getWI(a->shape[0], a->ctx->max_work_group_size);

	clSetKernelArg(a->ctx->kernels[16], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[16], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[16], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[16], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[16], 4, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[16], 5, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[16], 1, NULL, a->shape, &wi);

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
	wmatrix *b = wekuaMatrixCopy(a);

	clSetKernelArg(a->ctx->kernels[17], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[17], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[17], 2, 1, &b->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[17], 1, NULL, &a->size, a->work_items);

	wekuaMatrixSum(b, real, imag);

	double ang, n;

	if (a->com){
		if (real[0] == 0){
			ang = CL_M_PI_2;
		}else{
			ang = tanh(imag[0]/real[0])/2;	
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
	}

	wmatrix *b = wekuaMatrixCopy(a);
	wmatrix *c = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
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
		clSetKernelArg(a->ctx->kernels[19], 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(a->ctx->kernels[19], 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(a->ctx->kernels[19], 2, sizeof(cl_mem), &c->real);
		clSetKernelArg(a->ctx->kernels[19], 3, sizeof(cl_mem), &c->imag);
		clSetKernelArg(a->ctx->kernels[19], 4, 8, &k);
		clSetKernelArg(a->ctx->kernels[19], 5, 8, &b->shape[1]);
		clSetKernelArg(a->ctx->kernels[19], 6, 1, &b->com);

		clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[19], 1, NULL, a->shape, &a->work_items[3], we, befo, &event[k]);
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
	}

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
			clSetKernelArg(a->ctx->kernels[20], 0, sizeof(cl_mem), &b->real);
			clSetKernelArg(a->ctx->kernels[20], 1, sizeof(cl_mem), &b->imag);
			clSetKernelArg(a->ctx->kernels[20], 2, sizeof(cl_mem), &inv->real);
			clSetKernelArg(a->ctx->kernels[20], 3, sizeof(cl_mem), &inv->imag);
			clSetKernelArg(a->ctx->kernels[20], 4, 8, &k);
			clSetKernelArg(a->ctx->kernels[20], 5, 8, &a->shape[1]);
			clSetKernelArg(a->ctx->kernels[20], 6, 1, &a->com);
			clSetKernelArg(a->ctx->kernels[20], 7, 1, &otherm);
			clSetKernelArg(a->ctx->kernels[20], 8, 1, &t);
			clSetKernelArg(a->ctx->kernels[20], 9, 1, &otherm);

			clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[20], 1, NULL, a->shape, &a->work_items[3], we, befo, &event[t*(a->shape[1]-1)+k]);
			if (we == 0){
				we++;
			}
			befo = &event[t*(a->shape[1]-1)+k];
		}
	}
	clSetKernelArg(a->ctx->kernels[21], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[21], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[21], 2, sizeof(cl_mem), &inv->real);
	clSetKernelArg(a->ctx->kernels[21], 3, sizeof(cl_mem), &inv->imag);
	clSetKernelArg(a->ctx->kernels[21], 4, 8, &a->shape[1]);
	clSetKernelArg(a->ctx->kernels[21], 5, 1, &a->com);

	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[21], 2, NULL, a->shape, &a->work_items[1], 1, befo, &event[2*(a->shape[1]-1)]);

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
	uint32_t rang=0;
	wmatrix *b, *c;
	c = wekuaFillMatrix(a->ctx, a->shape[1], 1, 0.0, 0.0);
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
		clSetKernelArg(a->ctx->kernels[20], 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(a->ctx->kernels[20], 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(a->ctx->kernels[20], 2, sizeof(cl_mem), &nullptr);
		clSetKernelArg(a->ctx->kernels[20], 3, sizeof(cl_mem), &nullptr);
		clSetKernelArg(a->ctx->kernels[20], 4, 8, &k);
		clSetKernelArg(a->ctx->kernels[20], 5, 8, &a->shape[1]);
		clSetKernelArg(a->ctx->kernels[20], 6, 1, &a->com);
		clSetKernelArg(a->ctx->kernels[20], 7, 1, &otherm);
		clSetKernelArg(a->ctx->kernels[20], 8, 1, &t);
		clSetKernelArg(a->ctx->kernels[20], 9, 1, &otherm);

		clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[20], 1, NULL, &a->shape[d], &a->work_items[3+d], we, befo, &event[k]);
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

	clSetKernelArg(a->ctx->kernels[15], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[15], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[15], 2, sizeof(cl_mem), &c->real);
	clSetKernelArg(a->ctx->kernels[15], 3, sizeof(cl_mem), &c->imag);
	clSetKernelArg(a->ctx->kernels[15], 4, 8, &b->shape[1]);
	clSetKernelArg(a->ctx->kernels[15], 5, 1, &b->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[15], 2, NULL, a->shape, wi);
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
	if (a == NULL){
		return;
	}else if (a->com && real != NULL && imag != NULL){
		real[0] = a->raw_real[0]; imag[0] = a->raw_imag[0];
		for (uint64_t x=1; x < a->size; x++){
			if (sqrt(a->raw_real[x]*a->raw_real[x] + a->raw_imag[x]*a->raw_imag[x]) > sqrt(real[0]*real[0] + imag[0]*imag[0])){
				real[0] = a->raw_real[x];
				imag[0] = a->raw_imag[x];
			}
		}
	}else if (a->com == 0 && real != NULL){
		real[0] = a->raw_real[0];
		for (uint64_t x=1; x < a->size; x++){
			if (a->raw_real[x] > real[0]){
				real[0] = a->raw_real[x];
			}
		}
	}
}

void wekuaMatrixMin(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (a->com && real != NULL && imag != NULL){
		real[0] = a->raw_real[0]; imag[0] = a->raw_imag[0];
		for (uint64_t x=1; x < a->size; x++){
			if (sqrt(a->raw_real[x]*a->raw_real[x] + a->raw_imag[x]*a->raw_imag[x]) < sqrt(real[0]*real[0] + imag[0]*imag[0])){
				real[0] = a->raw_real[x];
				imag[0] = a->raw_imag[x];
			}
		}
	}else if (a->com == 0 && real != NULL){
		real[0] = a->raw_real[0];
		for (uint64_t x=1; x < a->size; x++){
			if (a->raw_real[x] < real[0]){
				real[0] = a->raw_real[x];
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
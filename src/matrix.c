#include "wekua.h"
#include <math.h>
#include <stdio.h>

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

void getLWI(void *x, void *y, uint32_t si, uint64_t max){
	uint64_t c = (uint64_t) pow(1.0*max, 1.0/si);
	for (uint32_t j=0; j<si; j++){
		if (((uint64_t*)x)[j] < c){
			((uint64_t*)y)[j] = ((uint64_t*)x)[j];
			continue;
		}
		((uint64_t*)y)[j] = c;
		while (((uint64_t*)x)[j]%((uint64_t*)y)[j] != 0){
			((uint64_t*)y)[j]--;
		}
	}
}

void MapBufferMatrix(wMatrix *a){
	a->raw_data = clEnqueueMapBuffer(a->ctx->command_queue, a->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);
}

void UnmapBufferMatrix(wMatrix *a){
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->data, a->raw_data, 0, NULL, NULL);
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
}

void wekuaMatrixPrint(wMatrix *a){
	if (a == NULL){
		return;
	}
	for (uint32_t y=0; y<a->r; y++){
		for (uint32_t x=0; x<a->c; x++){
			if (x == 0 && (y < 5 || y >= a->r-4)){
				printf("[");
			}
			if ((x < 4 || x >= a->c-4) && (y < 4 || y >= a->r-4)){
				printf("%14.5e", a->raw_data[y*a->c+x]);
			}else if ((x == 4 && (y < 5 || y >= a->r-4)) || (y == 4 && (x < 4 || x >= a->c-4))){
				printf("%14s", "...");
			}
			if (x == a->c-1 && (y < 5 || y >= a->r-4)){
				printf("]\n");
			}
		}
	}
	printf("\n");
}

wMatrix *wekuaAllocMatrix(wekuaContext *ctx, uint32_t r, uint32_t c, double alpha){
	if (r == 0 || c == 0 || ctx == NULL){
		return NULL;
	}
	int ret;
	wMatrix *Matrix = calloc(r*c, sizeof(wMatrix));
	Matrix->size = r*c;
	Matrix->data = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, Matrix->size*sizeof(double), NULL, &ret);
	if (ret != 0){
		printf("Failed to allocate new memory :-(\n");
		free(Matrix);
		return NULL;
	}
	Matrix->r = r;
	Matrix->c = c;
	Matrix->ctx = ctx;
	Matrix->work_items[0] = getWI(Matrix->size, ctx->max_work_item_sizes[0]);
	uint64_t shape[2];
	shape[0] = r;
	shape[1] = c;
	getLWI(shape, &Matrix->work_items[1], 2, ctx->max_work_group_size);
	clSetKernelArg(ctx->kernels[0], 0, sizeof(cl_mem), &Matrix->data);
	clSetKernelArg(ctx->kernels[0], 1, sizeof(double), &alpha);
	runKernel(ctx->command_queue, ctx->kernels[0], 1, NULL, &Matrix->size, Matrix->work_items);
	MapBufferMatrix(Matrix);
	return Matrix;
}

wMatrix *wekuaAllocMatrixRand(wekuaContext *ctx, uint32_t r, uint32_t c){
	wMatrix *a = wekuaAllocMatrix(ctx, r, c, 0.0);
	if (a == NULL){
		return NULL;
	}
	cl_mem rn_buf = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, a->size*8, NULL, NULL);
	int64_t *ran_buf = clEnqueueMapBuffer(ctx->command_queue, rn_buf, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, r*c*8, 0, 0, NULL, NULL);
	getRandomBuffer(ran_buf, a->size*8);
	clEnqueueUnmapMemObject(a->ctx->command_queue, rn_buf, ran_buf, 0, NULL, NULL);
	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[4], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[4], 1, sizeof(cl_mem), &rn_buf);
	runKernel(ctx->command_queue, ctx->kernels[4], 1, NULL, &a->size, a->work_items);
	clReleaseMemObject(rn_buf);
	MapBufferMatrix(a);
	return a;
}

wMatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint32_t r, uint32_t c, void *buf){
	wMatrix *a = wekuaAllocMatrix(ctx, r, c, 0.0);
	if (a == NULL){
		return NULL;
	}
	memcpy(a->raw_data, buf, a->size*sizeof(double));
	return a;
}

void wekuaFreeMatrix(wMatrix *Matrix){
	if (Matrix == NULL){
		return;
	}
	UnmapBufferMatrix(Matrix);
	clReleaseMemObject(Matrix->data);
	free(Matrix);
}

double wekuaMatrixGet(wMatrix *a, uint32_t x, uint32_t y){
	if (a == NULL){
		return 0.0;
	}
	if (x >= a->c || y >= a->r){
		return 0.0;
	}
	return a->raw_data[y*a->c+x];
}

void wekuaMatrixPut(wMatrix *a, uint32_t x, uint32_t y, double n){
	if (a == NULL){
		return;
	}
	if (x >= a->c || y >= a->r){
		return;
	}
	a->raw_data[y*a->c+x] = n;
}

wMatrix *wekuaMatrixCopy(wMatrix *a){
	wMatrix *b = wekuaAllocMatrix(a->ctx, a->r, a->c, 0.0);
	if (b == NULL){
		return NULL;
	}
	UnmapBufferMatrix(b);
	clSetKernelArg(a->ctx->kernels[3], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[3], 1, sizeof(cl_mem), &a->data);
	runKernel(a->ctx->command_queue, a->ctx->kernels[3], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(b);
	return b;
}

wMatrix *wekuaMatrixReshape(wMatrix *a, uint32_t r, uint32_t c){
	if (a == NULL){
		return NULL;
	}
	if (r*c != a->size){
		return NULL;
	}
	wMatrix *b = wekuaMatrixCopy(a);
	b->r = r;
	b->c = c;
	return b;
}

wMatrix *wekuaMatrixResize(wMatrix *a, uint32_t r, uint32_t c){
	if (r*c == 0){
		return NULL;
	}
	wMatrix *b = wekuaAllocMatrix(a->ctx, r, c, 0.0);
	UnmapBufferMatrix(b);
	uint64_t shape[2], wi[2];
	if (r < a->r){
		shape[0] = r;
	}else{
		shape[0] = a->r;
	}
	if (c < a->c){
		shape[1] = c;
	}else{
		shape[1] = a->c;
	}
	getLWI(shape, wi, 2, a->ctx->max_work_group_size);
	clSetKernelArg(a->ctx->kernels[15], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[15], 1, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[15], 2, sizeof(uint32_t), &a->c);
	clSetKernelArg(a->ctx->kernels[15], 3, sizeof(uint32_t), &b->c);
	runKernel(a->ctx->command_queue, a->ctx->kernels[15], 2, NULL, shape, wi);
	MapBufferMatrix(b);
	return b;

}

void wekuaMatrixAdd(wMatrix *a, wMatrix *b){
	if (a == NULL || b == NULL){
		return;
	}else if (a->r != b->r || a->c != b->c){
		return;
	}else if (a->ctx != b->ctx){
		return;
	}
	double alpha = 1;
	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[1], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[1], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[1], 2, sizeof(double), &alpha);
	runKernel(a->ctx->command_queue, a->ctx->kernels[1], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(a);
}

void wekuaMatrixAbs(wMatrix *a){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[9], 0, sizeof(cl_mem), a->data);
	runKernel(a->ctx->command_queue, a->ctx->kernels[9], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(a);
}

void wekuaMatrixSub(wMatrix *a, wMatrix *b){
	if (a == NULL || b == NULL){
		return;
	}else if (a->r != b->r && a->c != b->c){
		return;
	}else if (a->ctx != b->ctx){
		return;
	}
	double alpha = -1;
	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[1], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[1], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[1], 2, sizeof(double), &alpha);
	runKernel(a->ctx->command_queue, a->ctx->kernels[1], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(a);
}

void wekuaMatrixAbsdiff(wMatrix *a, wMatrix *b){
	if (a->ctx != b->ctx){
		return;
	}
	wekuaMatrixSub(a, b);
	wekuaMatrixAbs(a);
}

void wekuaMatrixDot(wMatrix *a, double alpha){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[2], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[2], 1, sizeof(double), &alpha);
	runKernel(a->ctx->command_queue, a->ctx->kernels[2], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(a);
}

wMatrix *wekuaMatrixTrans(wMatrix *a){
	if (a == NULL){
		return NULL;
	}
	wMatrix *b = wekuaAllocMatrix(a->ctx, a->c, a->r, 0.0);
	if (b == NULL){
		return NULL;
	}
	UnmapBufferMatrix(b);
	uint64_t shape[2];
	shape[0] = (uint64_t) a->r;
	shape[1] = (uint64_t) a->c;
	clSetKernelArg(a->ctx->kernels[5], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[5], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[5], 2, sizeof(uint32_t), &a->r);
	clSetKernelArg(a->ctx->kernels[5], 3, sizeof(uint32_t), &a->c);
	runKernel(a->ctx->command_queue, a->ctx->kernels[5], 2, NULL, shape, &a->work_items[1]);
	MapBufferMatrix(b);
	return b;
}

double wekuaMatrixSum(wMatrix *a){
	if (a == NULL){
		return 0.0;
	}
	double result;
	wMatrix *b, *c;
	b = wekuaAllocMatrix(a->ctx, a->r, 1, 0.0);
	c = wekuaAllocMatrix(a->ctx, 1, 1, 0.0);
	if (b == NULL || c == NULL){
		return 0.0;
	}
	uint64_t shape[2], wi[2];
	shape[0] = a->r;
	shape[1] = a->c;
	getLWI(shape, wi, 1, a->ctx->max_work_group_size);
	wi[1] = 1;
	UnmapBufferMatrix(b);
	UnmapBufferMatrix(c);
	clSetKernelArg(a->ctx->kernels[10], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[10], 1, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[10], 2, sizeof(uint32_t), &a->c);
	runKernel(a->ctx->command_queue, a->ctx->kernels[10], 2, NULL, shape, wi);
	shape[0] = 1;
	shape[1] = b->r;
	wi[0] = 1;
	clSetKernelArg(a->ctx->kernels[10], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[10], 1, sizeof(cl_mem), &c->data);
	clSetKernelArg(a->ctx->kernels[10], 2, sizeof(uint32_t), &b->r);
	runKernel(b->ctx->command_queue, b->ctx->kernels[10], 2, NULL, shape, wi);
	MapBufferMatrix(c);
	result = c->raw_data[0];
	wekuaFreeMatrix(b);
	wekuaFreeMatrix(c);
	return result;
}

double wekuaMatrixMul(wMatrix *a){
	if (a == NULL){
		return 0.0;
	}
	double result;
	wMatrix *b, *c;
	b = wekuaAllocMatrix(a->ctx, a->r, 1, 1.0);
	c = wekuaAllocMatrix(a->ctx, 1, 1, 1.0);
	if (b == NULL || c == NULL){
		return 0.0;
	}
	uint64_t shape[2], wi[2];
	shape[0] = a->r;
	shape[1] = a->c;
	wi[0] = a->work_items[1];
	wi[1] = 1;
	UnmapBufferMatrix(b);
	UnmapBufferMatrix(c);
	clSetKernelArg(a->ctx->kernels[11], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[11], 1, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[11], 2, sizeof(uint32_t), &a->c);
	runKernel(a->ctx->command_queue, a->ctx->kernels[11], 2, NULL, shape, wi);
	shape[0] = 1;
	shape[1] = b->r;
	wi[0] = 1;
	clSetKernelArg(a->ctx->kernels[11], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[11], 1, sizeof(cl_mem), &c->data);
	clSetKernelArg(a->ctx->kernels[11], 2, sizeof(uint32_t), &b->r);
	runKernel(b->ctx->command_queue, b->ctx->kernels[11], 2, NULL, shape, wi);
	MapBufferMatrix(c);
	result = c->raw_data[0];
	wekuaFreeMatrix(b);
	wekuaFreeMatrix(c);
	return result;
}

double wekuaMatrixMean(wMatrix *a){
	double total = wekuaMatrixSum(a);
	return total/a->size;
}

wMatrix *wekuaMatrixIden(wekuaContext *ctx, uint32_t c){
	wMatrix *a = wekuaAllocMatrix(ctx, c, c, 0.0);
	if (a == NULL){
		return NULL;
	}
	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[6], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[6], 1, sizeof(uint32_t), &c);
	runKernel(a->ctx->command_queue, a->ctx->kernels[6], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(a);
	return a;
}

wMatrix *wekuaSubMatrix(wMatrix *a, uint32_t x, uint32_t w, uint32_t y, uint32_t h){
	if (a == NULL){
		return NULL;
	}
	if (x >= a->c || y >= a->r){
		return NULL;
	}
	wMatrix *b = wekuaAllocMatrix(a->ctx, h, w, 0.0);
	if (b == NULL){
		return NULL;
	}
	uint64_t shape[2];
	shape[0] = (uint64_t) b->r;
	shape[1] = (uint64_t) b->c;
	UnmapBufferMatrix(b);
	clSetKernelArg(a->ctx->kernels[7], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[7], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[7], 2, sizeof(uint32_t), &x);
	clSetKernelArg(a->ctx->kernels[7], 3, sizeof(uint32_t), &y);
	clSetKernelArg(a->ctx->kernels[7], 4, sizeof(uint32_t), &w);
	clSetKernelArg(a->ctx->kernels[7], 5, sizeof(uint32_t), &a->c);
	runKernel(a->ctx->command_queue, a->ctx->kernels[7], 2, NULL, shape, &a->work_items[1]);
	MapBufferMatrix(b);
	return b;
}

wMatrix *wekuaMatrixProduct(wMatrix *a, wMatrix *b){
	if (a == NULL || b == NULL){
		return NULL;
	}else if (a->c != b->r){
		return NULL;
	}else if (a->ctx != b->ctx){
		return NULL;
	}
	wMatrix *c = wekuaAllocMatrix(a->ctx, a->r, b->c, 0.0);
	if (c == NULL){
		return NULL;
	}
	uint64_t shape[3], wi[3];
	shape[0] = (uint64_t) a->r; // r
	shape[1] = (uint64_t) a->c; // c
	shape[2] = (uint64_t) b->c; // k
	getLWI(shape, wi, 3, a->ctx->max_work_group_size);
	UnmapBufferMatrix(c);
	clSetKernelArg(c->ctx->kernels[8], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(c->ctx->kernels[8], 1, sizeof(cl_mem), &b->data);
	clSetKernelArg(c->ctx->kernels[8], 2, sizeof(cl_mem), &c->data);
	clSetKernelArg(c->ctx->kernels[8], 3, sizeof(uint32_t), &a->c);
	clSetKernelArg(c->ctx->kernels[8], 4, sizeof(uint32_t), &b->c);
	runKernel(c->ctx->command_queue, c->ctx->kernels[8], 3, NULL, shape, wi);
	MapBufferMatrix(c);
	return c;
}

wMatrix *wekuaMatrixInv(wMatrix *a){
	if (wekuaMatrixDet(a) == 0.0){
		return NULL;
	}
	wMatrix *b = wekuaMatrixCopy(a);
	wMatrix *i = wekuaMatrixIden(a->ctx, a->c);
	UnmapBufferMatrix(b);
	UnmapBufferMatrix(i);
	uint64_t shape[2], wi[2];
	shape[0] = a->r;
	shape[1] = a->r;
	wi[0] = 1;
	wi[1] = a->work_items[1];
	for (uint32_t t=0; t<2; t++){
		clSetKernelArg(a->ctx->kernels[12], 0, sizeof(cl_mem), &b->data);
		clSetKernelArg(a->ctx->kernels[12], 1, sizeof(cl_mem), &i->data);
		clSetKernelArg(a->ctx->kernels[12], 2, sizeof(uint32_t), &t);
		clSetKernelArg(a->ctx->kernels[12], 3, sizeof(uint32_t), &b->r);
		runKernel(b->ctx->command_queue, b->ctx->kernels[12], 2, NULL, shape, wi);
	}
	clSetKernelArg(a->ctx->kernels[13], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[13], 1, sizeof(cl_mem), &i->data);
	clSetKernelArg(a->ctx->kernels[13], 2, sizeof(uint32_t), &b->r);
	runKernel(b->ctx->command_queue, b->ctx->kernels[13], 2, NULL, shape, &a->work_items[1]);
	wekuaFreeMatrix(b);
	MapBufferMatrix(i);
	return i;
}

double wekuaMatrixDet(wMatrix *a){
	if (a->r != a->c){
		return 0.0;
	}
	double det;
	if (a->r == 1){
		det = a->raw_data[0];
	}else{
		wMatrix *b = wekuaMatrixCopy(a);
		wMatrix *c = wekuaAllocMatrix(a->ctx, a->r, a->r, 1.0);
		UnmapBufferMatrix(b);
		UnmapBufferMatrix(c);
		uint64_t shape[2], wi[2];
		shape[0] = a->r;
		shape[1] = a->r;
		wi[0] = 1;
		wi[1] = a->work_items[1];
		clSetKernelArg(a->ctx->kernels[14], 0, sizeof(cl_mem), &b->data);
		clSetKernelArg(a->ctx->kernels[14], 1, sizeof(cl_mem), &c->data);
		clSetKernelArg(a->ctx->kernels[14], 2, sizeof(uint32_t), &a->c);
		runKernel(a->ctx->command_queue, a->ctx->kernels[14], 2, NULL, shape, wi);
		MapBufferMatrix(b);
		MapBufferMatrix(c);
		det = wekuaMatrixMul(c);
		wekuaFreeMatrix(b);
		wekuaFreeMatrix(c);
	}
	return det;
}

wMatrix *wekuaMatrixSolve(wMatrix *a, wMatrix *b){
	if (a->r != a->c){
		return NULL;
	}
	wMatrix *ia = wekuaMatrixInv(a);
	wMatrix *c = wekuaMatrixProduct(ia, b);
	wekuaFreeMatrix(ia);
	return c;
}

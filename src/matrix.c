#include "wekua.h"
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

wMatrix *wekuaAllocMatrix(wekuaContext *ctx, uint32_t r, uint32_t c, double alpha){
	wMatrix *Matrix = calloc(r*c, sizeof(wMatrix));
	Matrix->size = r*c;
	Matrix->data = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, r*c*sizeof(double), NULL, NULL);
	Matrix->r = r;
	Matrix->c = c;
	Matrix->ctx = ctx;
	Matrix->work_items[0] = getWI(Matrix->size, ctx->max_work_item_sizes[0]);
	Matrix->work_items[1] = getWI(Matrix->r, ctx->max_work_item_sizes[0]);
	Matrix->work_items[2] = getWI(Matrix->c, ctx->max_work_item_sizes[1]);
	clSetKernelArg(ctx->kernels[0], 0, sizeof(cl_mem), &Matrix->data);
	clSetKernelArg(ctx->kernels[0], 1, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernels[0], 1, NULL, &Matrix->size, Matrix->work_items, 0, NULL, NULL);
	Matrix->raw_data = clEnqueueMapBuffer(ctx->command_queue, Matrix->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, r*c*sizeof(double), 0, 0, NULL, NULL);
	return Matrix;
}

wMatrix *wekuaAllocMatrixRand(wekuaContext *ctx, uint32_t r, uint32_t c){
	wMatrix *a = wekuaAllocMatrix(ctx, r, c, 0.0);
	cl_mem rn_buf = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, a->size*8, NULL, NULL);
	int64_t *ran_buf = clEnqueueMapBuffer(ctx->command_queue, rn_buf, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, r*c*8, 0, 0, NULL, NULL);
	getRandomBuffer(ran_buf, a->size*8);
	clEnqueueUnmapMemObject(a->ctx->command_queue, rn_buf, ran_buf, 0, NULL, NULL);
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->data, a->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[4], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[4], 1, sizeof(cl_mem), &rn_buf);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[4], 1, NULL, &a->size, a->work_items, 0, NULL, NULL);
	clReleaseMemObject(rn_buf);
	a->raw_data = clEnqueueMapBuffer(ctx->command_queue, a->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, r*c*sizeof(double), 0, 0, NULL, NULL);
	return a;
}

wMatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint32_t r, uint32_t c, void *buf){
	wMatrix *a = wekuaAllocMatrix(ctx, r, c, 0.0);
	memcpy(a->raw_data, buf, r*c*sizeof(double));
	return a;
}

void wekuaFreeMatrix(wMatrix *Matrix){
	clEnqueueUnmapMemObject(Matrix->ctx->command_queue, Matrix->data, Matrix->raw_data, 0, NULL, NULL);
	clReleaseMemObject(Matrix->data);
	free(Matrix);
}

double wekuaMatrixGet(wMatrix *a, uint32_t x, uint32_t y){
	return a->raw_data[y*a->c+x];
}

void wekuaMatrixPut(wMatrix *a, uint32_t x, uint32_t y, double n){
	a->raw_data[y*a->c+x] = n;
}

wMatrix *wekuaMatrixCopy(wMatrix *a){
	wMatrix *b = wekuaAllocMatrix(a->ctx, a->r, a->c, 0.0);
	clEnqueueUnmapMemObject(b->ctx->command_queue, b->data, b->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[3], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[3], 1, sizeof(cl_mem), &a->data);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[3], 1, NULL, &a->size, a->work_items, 0, NULL, NULL);
	b->raw_data = clEnqueueMapBuffer(b->ctx->command_queue, b->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->r*a->c*sizeof(double), 0, 0, NULL, NULL);
	return b;
}

void wekuaMatrixAdd(wMatrix *a, wMatrix *b){
	if (a->r != b->r || a->c != b->c){
		return;
	}
	double alpha = 1;
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->data, a->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[1], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[1], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[1], 2, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[1], 1, NULL, &a->size, a->work_items, 0, NULL, NULL);
	a->raw_data = clEnqueueMapBuffer(a->ctx->command_queue, a->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->r*a->c*sizeof(double), 0, 0, NULL, NULL);
}

void wekuaMatrixSub(wMatrix *a, wMatrix *b){
	if (a->r != a->r && a->c != a->c){
		return;
	}
	double alpha = -1;
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->data, a->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[1], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[1], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[1], 2, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[1], 1, NULL, &a->size, a->work_items, 0, NULL, NULL);
	a->raw_data = clEnqueueMapBuffer(a->ctx->command_queue, a->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->r*a->c*sizeof(double), 0, 0, NULL, NULL);
}

void wekuaMatrixDot(wMatrix *a, double alpha){
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->data, a->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[2], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[2], 1, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[2], 1, NULL, &a->size, a->work_items, 0, NULL, NULL);
	a->raw_data = clEnqueueMapBuffer(a->ctx->command_queue, a->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->r*a->c*sizeof(double), 0, 0, NULL, NULL);
}

wMatrix *wekuaMatrixTrans(wMatrix *a){
	wMatrix *b = wekuaAllocMatrix(a->ctx, a->c, a->r, 0.0);
	uint64_t shape[2];
	shape[0] = (uint64_t) a->r;
	shape[1] = (uint64_t) a->c;
	clSetKernelArg(a->ctx->kernels[5], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[5], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[5], 2, sizeof(uint32_t), &a->r);
	clSetKernelArg(a->ctx->kernels[5], 3, sizeof(uint32_t), &a->c);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[5], 2, NULL, shape, &a->work_items[1], 0, NULL, NULL);
	b->raw_data = clEnqueueMapBuffer(b->ctx->command_queue, b->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->r*a->c*sizeof(double), 0, 0, NULL, NULL);
	return b;
}

wMatrix *wekuaMatrixIden(wekuaContext *ctx, uint32_t c){
	wMatrix *a = wekuaAllocMatrix(ctx, c, c, 0.0);
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->data, a->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[6], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[6], 1, sizeof(uint32_t), &c);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[6], 1, NULL, &a->size, a->work_items, 0, NULL, NULL);
	a->raw_data = clEnqueueMapBuffer(a->ctx->command_queue, a->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->r*a->c*sizeof(double), 0, 0, NULL, NULL);
	return a;
}

wMatrix *wekuaSubMatrix(wMatrix *a, uint32_t x, uint32_t w, uint32_t y, uint32_t h){
	wMatrix *b = wekuaAllocMatrix(a->ctx, h, w, 0.0);
	uint64_t shape[2];
	shape[0] = (uint64_t) b->r;
	shape[1] = (uint64_t) b->c;
	clEnqueueUnmapMemObject(b->ctx->command_queue, b->data, b->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[7], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[7], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[7], 2, sizeof(uint32_t), &x);
	clSetKernelArg(a->ctx->kernels[7], 3, sizeof(uint32_t), &y);
	clSetKernelArg(a->ctx->kernels[7], 4, sizeof(uint32_t), &w);
	clSetKernelArg(a->ctx->kernels[7], 5, sizeof(uint32_t), &a->c);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[7], 2, NULL, shape, &b->work_items[1], 0, NULL, NULL);
	b->raw_data = clEnqueueMapBuffer(b->ctx->command_queue, b->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, b->r*b->c*sizeof(double), 0, 0, NULL, NULL);
	return b;
}

wMatrix *wekuaMatrixProduct(wMatrix *a, wMatrix *b){
	if (a->c != b->r){
		return NULL;
	}
	wMatrix *c = wekuaAllocMatrix(a->ctx, a->r, b->c, 0.0);
	uint64_t shape[3];
	shape[0] = (uint64_t) a->r;
	shape[1] = (uint64_t) a->c;
	shape[2] = (uint64_t) b->c;
	uint64_t wi[3];
	memcpy(wi, &a->work_items[1], 2*8);
	wi[2] = getWI(b->c, b->ctx->max_work_item_sizes[2]);
	clEnqueueUnmapMemObject(c->ctx->command_queue, c->data, c->raw_data, 0, NULL, NULL);
	clSetKernelArg(c->ctx->kernels[8], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(c->ctx->kernels[8], 1, sizeof(cl_mem), &b->data);
	clSetKernelArg(c->ctx->kernels[8], 2, sizeof(cl_mem), &c->data);
	clSetKernelArg(c->ctx->kernels[8], 3, sizeof(uint32_t), &a->r);
	clSetKernelArg(c->ctx->kernels[8], 4, sizeof(uint32_t), &a->c);
	clSetKernelArg(c->ctx->kernels[8], 5, sizeof(uint32_t), &b->c);
	clEnqueueNDRangeKernel(c->ctx->command_queue, c->ctx->kernels[8], 3, NULL, shape, wi, 0, NULL, NULL);
	c->raw_data = clEnqueueMapBuffer(c->ctx->command_queue, c->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, c->r*c->c*sizeof(double), 0, 0, NULL, NULL);
	return c;
}

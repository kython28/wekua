#include "wekua.h"

#include <stdio.h>

uint64_t getWI(uint64_t a, uint64_t max){
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
	Matrix->work_items = Matrix->size;
	if (Matrix->work_items > ctx->max_work_item_sizes[0]){
		Matrix->work_items = getWI(Matrix->work_items, ctx->max_work_item_sizes[0]);
	}
	clSetKernelArg(ctx->kernels[0], 0, sizeof(cl_mem), &Matrix->data);
	clSetKernelArg(ctx->kernels[0], 1, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernels[0], 1, NULL, &Matrix->size, &Matrix->work_items, 0, NULL, NULL);
	Matrix->raw_data = clEnqueueMapBuffer(ctx->command_queue, Matrix->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, r*c*sizeof(double), 0, 0, NULL, NULL);
	return Matrix;
}

void wekuaFreeMatrix(wMatrix *Matrix){
	clEnqueueUnmapMemObject(Matrix->ctx->command_queue, Matrix->data, Matrix->raw_data, 0, NULL, NULL);
	clReleaseMemObject(Matrix->data);
	free(Matrix);
}

wMatrix *wekuaMatrixCopy(wMatrix *a){
	wMatrix *b = wekuaAllocMatrix(a->ctx, a->r, a->c, 0.0);
	clEnqueueUnmapMemObject(b->ctx->command_queue, b->data, b->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[3], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(a->ctx->kernels[3], 1, sizeof(cl_mem), &a->data);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[3], 1, NULL, &a->size, &a->work_items, 0, NULL, NULL);
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
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[1], 1, NULL, &a->size, &a->work_items, 0, NULL, NULL);
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
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[1], 1, NULL, &a->size, &a->work_items, 0, NULL, NULL);
	a->raw_data = clEnqueueMapBuffer(a->ctx->command_queue, a->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->r*a->c*sizeof(double), 0, 0, NULL, NULL);
}

void wekuaMatrixDot(wMatrix *a, double alpha){
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->data, a->raw_data, 0, NULL, NULL);
	clSetKernelArg(a->ctx->kernels[2], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(a->ctx->kernels[2], 1, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(a->ctx->command_queue, a->ctx->kernels[2], 1, NULL, &a->size, &a->work_items, 0, NULL, NULL);
	a->raw_data = clEnqueueMapBuffer(a->ctx->command_queue, a->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->r*a->c*sizeof(double), 0, 0, NULL, NULL);
}

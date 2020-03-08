#include "wekua.h"

uint64_t getShLe(uint32_t *shape, uint32_t dim){
	uint64_t n = 1;
	for (uint32_t x=0; x<dim; x++){
		n *= shape[x];
	}
	return n;
}

wTensor *wekuaAllocTensor(wekuaContext *ctx, uint32_t dim, uint32_t *shape, double alpha){
	uint64_t size = getShLe(shape, dim);
	wTensor *tensor = calloc(size, sizeof(wTensor));
	tensor->size = size;
	tensor->data = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, size*sizeof(double), NULL, NULL);
	tensor->dim = dim;
	tensor->shape = (uint32_t*) calloc(dim, 4);
	memcpy(tensor->shape, shape, dim*4);
	tensor->raw_data = clEnqueueMapBuffer(ctx->command_queue, tensor->data, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, size, 0, 0, NULL, NULL);
	uint64_t work_items = tensor->size;
	if (work_items > ctx->max_work_item_sizes[0]){
		work_items = ctx->max_work_item_sizes[0];
	}
	clSetKernelArg(ctx->kernels[0], 0, sizeof(cl_mem), &tensor->data);
	clSetKernelArg(ctx->kernels[0], 1, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernels[0], 1, NULL, &tensor->size, &work_items, 0, NULL, NULL);
	return tensor;
}

void wekuaFreeTensor(wekuaContext *ctx, wTensor *tensor){
	clEnqueueUnmapMemObject(ctx->command_queue, tensor->data, tensor->raw_data, 0, NULL, NULL);
	clReleaseMemObject(tensor->data);
	free(tensor->shape);
	free(tensor);
}

wTensor *wekuaTensorCopy(wekuaContext *ctx, wTensor *a){
	wTensor *b = wekuaAllocTensor(ctx, a->dim, a->shape, 0.0);
	double alpha = 0;
	uint64_t work_items = a->size;
	if (work_items > ctx->max_work_item_sizes[0]){
		work_items = ctx->max_work_item_sizes[0];
	}
	clSetKernelArg(ctx->kernels[1], 0, sizeof(cl_mem), &b->data);
	clSetKernelArg(ctx->kernels[1], 1, sizeof(cl_mem), &a->data);
	clSetKernelArg(ctx->kernels[1], 2, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernels[1], 1, NULL, &a->size, &work_items, 0, NULL, NULL);
	return b;
}

void wekuaTensorAdd(wekuaContext *ctx, wTensor *a, wTensor *b){
	double alpha = 1;
	uint64_t work_items = a->size;
	if (work_items > ctx->max_work_item_sizes[0]){
		work_items = ctx->max_work_item_sizes[0];
	}
	clSetKernelArg(ctx->kernels[1], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(ctx->kernels[1], 1, sizeof(cl_mem), &b->data);
	clSetKernelArg(ctx->kernels[1], 2, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernels[1], 1, NULL, &a->size, &work_items, 0, NULL, NULL);
}

void wekuaTensorSub(wekuaContext *ctx, wTensor *a, wTensor *b){
	double alpha = -1;
	uint64_t work_items = a->size;
	if (work_items > ctx->max_work_item_sizes[0]){
		work_items = ctx->max_work_item_sizes[0];
	}
	clSetKernelArg(ctx->kernels[1], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(ctx->kernels[1], 1, sizeof(cl_mem), &b->data);
	clSetKernelArg(ctx->kernels[1], 2, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernels[1], 1, NULL, &a->size, &work_items, 0, NULL, NULL);
}

void wekuaTensorDot(wekuaContext *ctx, wTensor *a, double alpha){
	uint64_t work_items = a->size;
	if (work_items > ctx->max_work_item_sizes[0]){
		work_items = ctx->max_work_item_sizes[0];
	}
	clSetKernelArg(ctx->kernels[0], 0, sizeof(cl_mem), &a->data);
	clSetKernelArg(ctx->kernels[0], 1, sizeof(double), &alpha);
	clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernels[0], 1, NULL, &a->size, &work_items, 0, NULL, NULL);
}
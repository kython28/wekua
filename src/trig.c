#include "wekua.h"

void wTrig(wmatrix *a, uint32_t kn, uint32_t nw, cl_event *be, cl_event *e){

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[kn];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 8, &a->real_size[1]);
	clSetKernelArg(kernel, 3, 1, &a->com);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1], nw, be, e);
}

void wekuaMatrixSin(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}
	wTrig(a, 5, nw, be, e);
}

void wekuaMatrixCos(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}
	wTrig(a, 6, nw, be, e);
}

void wekuaMatrixTan(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}
	wTrig(a, 7, nw, be, e);
}

void wekuaMatrixSinh(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}
	wTrig(a, 8, nw, be, e);
}

void wekuaMatrixCosh(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}
	wTrig(a, 9, nw, be, e);
}

void wekuaMatrixTanh(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}
	wTrig(a, 10, nw, be, e);
}
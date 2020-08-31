#include "wekua.h"

void runKernel(cl_command_queue cmd, cl_kernel kernel, uint32_t ndim, uint64_t *offsi, uint64_t *glosi, uint64_t *losi);

void wTrig(wmatrix *a, uint32_t kn){

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[kn];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 1, &a->com);

	runKernel(ctx->command_queue, kernel, 1, NULL, &a->size, a->work_items);
}

void wekuaMatrixSin(wmatrix *a){
	if (a == NULL){
		return;
	}
	wTrig(a, 5);
}

void wekuaMatrixCos(wmatrix *a){
	if (a == NULL){
		return;
	}
	wTrig(a, 6);
}

void wekuaMatrixTan(wmatrix *a){
	if (a == NULL){
		return;
	}
	wTrig(a, 7);
}

void wekuaMatrixSinh(wmatrix *a){
	if (a == NULL){
		return;
	}
	wTrig(a, 8);
}

void wekuaMatrixCosh(wmatrix *a){
	if (a == NULL){
		return;
	}
	wTrig(a, 9);
}

void wekuaMatrixTanh(wmatrix *a){
	if (a == NULL){
		return;
	}
	wTrig(a, 10);
}
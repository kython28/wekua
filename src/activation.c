#include "wekua.h"

void runKernel(cl_command_queue cmd, cl_kernel kernel, uint32_t ndim, uint64_t *offsi, uint64_t *glosi, uint64_t *losi);
void acti(wmatrix *a, uint32_t id){
	if (a == NULL){
		return;
	}

	clSetKernelArg(a->ctx->kernels[id], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[id], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[id], 2, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[id], 1, NULL, &a->size, a->work_items);
}

void wekuaHardlim(wmatrix *a){
	acti(a, 24);
}

void wekuaHardlims(wmatrix *a){
	acti(a, 25);
}

void wekuaSatlin(wmatrix *a){
	acti(a, 26);
}

void wekuaSatlins(wmatrix *a){
	acti(a, 27);
}

void wekuaSigmoid(wmatrix *a){
	acti(a, 28);
}

void wekuaTanh(wmatrix *a){
	wekuaMatrixTanh(a);
}

void wekuaReLU(wmatrix *a){
	acti(a, 29);
}

void wekuaLeakyReLU(wmatrix *a){
	acti(a, 30);
}
void wekuaSoftplus(wmatrix *a){
	acti(a, 31);
	WekuaMatrixLn(a);
}
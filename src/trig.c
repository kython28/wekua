#include "wekua.h"

uint64_t getWI(uint64_t a, uint64_t max);
void getLWI(void *x, void *y, uint32_t si, uint64_t max);
void MapBufferMatrix(wmatrix *a);
void UnmapBufferMatrix(wmatrix *a);
void runKernel(cl_command_queue cmd, cl_kernel kernel, uint32_t ndim, uint64_t *offsi, uint64_t *glosi, uint64_t *losi);

void wTrig(wmatrix *a, uint32_t kn){
	clSetKernelArg(a->ctx->kernels[kn], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[kn], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[kn], 2, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[kn], 1, NULL, &a->size, a->work_items);
}

void wekuaMatrixSin(wmatrix *a){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	wTrig(a, 7);
	MapBufferMatrix(a);
}

void wekuaMatrixCos(wmatrix *a){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	wTrig(a, 8);
	MapBufferMatrix(a);
}

void wekuaMatrixTan(wmatrix *a){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	wTrig(a, 9);
	MapBufferMatrix(a);
}

void wekuaMatrixSinh(wmatrix *a){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	wTrig(a, 10);
	MapBufferMatrix(a);
}

void wekuaMatrixCosh(wmatrix *a){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	wTrig(a, 11);
	MapBufferMatrix(a);
}

void wekuaMatrixTanh(wmatrix *a){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	wTrig(a, 12);
	MapBufferMatrix(a);
}
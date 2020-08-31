#include "wekua.h"

void runKernel(cl_command_queue cmd, cl_kernel kernel, uint32_t ndim, uint64_t *offsi, uint64_t *glosi, uint64_t *losi);
uint64_t getCol(wmatrix *a);

void acti(wmatrix *a, uint32_t id){
	if (a == NULL){
		return;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[id];
	uint64_t col = getCol(a);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 8, &col);
	clSetKernelArg(kernel, 3, 1, &a->com);

	runKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1]);
}




// Linear activation function

void flinear(void *data, wmatrix *a){
	return;
}

wmatrix *fdlinear(void *data, wmatrix *a){
	return wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
}

wacti *wekuaFLinear(){
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->acti = &flinear;
	a->acti_dev = &fdlinear;
	return a;
}




// Sigmoid activation function

void fsigmoid(void *data, wmatrix *a){
	acti(a, 22);
}

wmatrix *fdsigmoid(void *data, wmatrix *a){
	wmatrix *b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
	wekuaMatrixSub(b, a);
	wekuaMatrixDot(b, a);
	return b;
}

wacti *wekuaSigmoid(){
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->acti = &fsigmoid;
	a->acti_dev = &fdsigmoid;
	return a;
}




// Tanh activation function

void ftanh(void *data, wmatrix *a){
	wekuaMatrixTanh(a);
}

wmatrix *fdtanh(void *data, wmatrix *a){
	wmatrix *b, *c;
	b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
	c = wekuaMatrixCopy(a);
	wekuaMatrixDot(c, a);
	wekuaMatrixSub(b, c);
	wekuaFreeMatrix(c);
	return b;
}

wacti *wekuaTanh(){
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->acti = &ftanh;
	a->acti_dev = &fdtanh;
	return a;
}



// ReLU activation function

void frelu(void *data, wmatrix *a){
	acti(a, 29);
}

wmatrix *fdrelu(void *data, wmatrix *a){
	wmatrix *b = wekuaMatrixCopy(a);
	acti(b, 30);
	return b;
}

wacti *wekuaReLU(){
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->acti = &frelu;
	a->acti_dev = &fdrelu;
	return a;
}




// LeakyReLU activation function

void flrelu(void *data, wmatrix *a){
	if (a == NULL){
		return;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[23];
	uint64_t col = getCol(a);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 8, &col);
	clSetKernelArg(kernel, 3, sizeof(double), data);
	clSetKernelArg(kernel, 4, 1, &a->com);

	runKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1]);
}

wmatrix *fdlrelu(void *data, wmatrix *a){
	wmatrix *b = wekuaMatrixCopy(a);
	wekuaMatrixDivide(b, a);
	frelu(data, b);
	return b;
}

wacti *wekuaLeakyReLU(double alpha){
	if (alpha == 0.0){
		return NULL;
	}
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->data = calloc(1, sizeof(double));
	((double*)a->data)[0] = alpha;
	a->acti = &flrelu;
	a->acti_dev = &fdrelu;
	return a;
}



void runWekuaActi(wacti *a, wmatrix *b){
	if (a == NULL || b == NULL){
		return;
	}
	a->acti(a->data, b);
}

wmatrix *getDevWekuaActi(wacti *a, wmatrix *b){
	if (a == NULL || b == NULL){
		return NULL;
	}
	return a->acti_dev(a->data, b);
}

void wekuaFreeActi(wacti *a){
	if (a == NULL){
		return;
	}else if (a->data != NULL){
		free(a->data);
	}
	free(a);
}
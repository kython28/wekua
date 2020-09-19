#include "wekua.h"

void acti(wmatrix *a, uint32_t id, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[id];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 8, &a->real_size[1]);
	clSetKernelArg(kernel, 3, 1, &a->com);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1], nw, be, e);
}


// Linear activation function

void flinear(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	return;
}

wmatrix *fdlinear(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	return wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
}

wacti *wekuaFLinear(){
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->acti = &flinear;
	a->acti_dev = &fdlinear;
	return a;
}




// Sigmoid activation function

void fsigmoid(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	cl_event e;
	acti(a, 21, nw, be, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);
}

wmatrix *fdsigmoid(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	wmatrix *b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
	cl_event e[2];
	wekuaMatrixSub(b, a, nw, be, e);
	wekuaMatrixDot(b, a, 1, e, &e[1]);
	clWaitForEvents(1, &e[1]);

	clReleaseEvent(e[0]);
	clReleaseEvent(e[1]);

	return b;
}

wacti *wekuaSigmoid(){
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->acti = &fsigmoid;
	a->acti_dev = &fdsigmoid;
	return a;
}




// Tanh activation function

void ftanh(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	cl_event e;
	wekuaMatrixTanh(a, nw, be, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);
}

wmatrix *fdtanh(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	wmatrix *b, *c;

	cl_event e[3];

	b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
	c = wekuaMatrixCopy(a, nw, be, e);
	wekuaMatrixDot(c, a, 1, e, &e[1]);
	wekuaMatrixSub(b, c, 1, &e[1], &e[2]);
	wekuaFreeMatrix(c, 1, &e[2]);

	for (uint32_t j=0; j<3; j++) clReleaseEvent(e[j]);

	return b;
}

wacti *wekuaTanh(){
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->acti = &ftanh;
	a->acti_dev = &fdtanh;
	return a;
}



// ReLU activation function

void frelu(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	cl_event e;
	acti(a, 28, nw, be, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);
}

wmatrix *fdrelu(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	cl_event e[2];

	wmatrix *b = wekuaMatrixCopy(a, nw, be, e);
	acti(b, 29, 1, e, &e[1]);

	clWaitForEvents(1, &e[1]);
	clReleaseEvent(e[0]);
	clReleaseEvent(e[1]);
	return b;
}

wacti *wekuaReLU(){
	wacti *a = (wacti*) calloc(1, sizeof(wacti));
	a->acti = &frelu;
	a->acti_dev = &fdrelu;
	return a;
}




// LeakyReLU activation function

void flrelu(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	if (a == NULL){
		return;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[23];
	cl_event e;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 8, &a->real_size[1]);
	clSetKernelArg(kernel, 3, sizeof(double), data);
	clSetKernelArg(kernel, 4, 1, &a->com);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, a->offset, a->shape, &a->work_items[1], nw, be, &e);
	clWaitForEvents(1, &e);
	clReleaseEvent(e);
}

wmatrix *fdlrelu(void *data, wmatrix *a, uint32_t nw, cl_event *be){
	cl_event e[3];
	wmatrix *b, *c;
	b = wekuaMatrixCopy(a, nw, be, e);
	c = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], CL_DBL_EPSILON, 0.0);
	wekuaMatrixAdd(b, c, 1, e, &e[1]);
	wekuaMatrixDivide(b, b, 1, &e[1], &e[2]);
	frelu(data, b, 1, &e[2]);

	clReleaseEvent(e[0]);
	clReleaseEvent(e[1]);
	clReleaseEvent(e[2]);
	wekuaFreeMatrix(c, 0, NULL);

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



void runWekuaActi(wacti *a, wmatrix *b, uint32_t nw, cl_event *be){
	if (a == NULL || b == NULL){
		return;
	}
	a->acti(a->data, b, nw, be);
}

wmatrix *getDevWekuaActi(wacti *a, wmatrix *b, uint32_t nw, cl_event *be){
	if (a == NULL || b == NULL){
		return NULL;
	}
	return a->acti_dev(a->data, b, nw, be);
}

void wekuaFreeActi(wacti *a, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, NULL);
	if (a == NULL){
		return;
	}else if (a->data != NULL){
		free(a->data);
	}
	free(a);
}
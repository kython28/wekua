#include "wekua.h"
#include <math.h>

void wekuaBlasAxpy(double alpha, double beta, wmatrix *x, wmatrix *y, uint32_t nw, cl_event *be, cl_event *e){
	if (x == NULL || y == NULL){
		return;
	}
	wekuaContext *ctx = x->ctx;
	cl_kernel kernel = ctx->kernels[3];

	if (x->com || y->com || beta != 0.0){
		if (createComplexMatrix(y) || createComplexMatrix(x)){
			return;
		}
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &x->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &y->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &y->imag);
	clSetKernelArg(kernel, 4, 1, &x->com);
	clSetKernelArg(kernel, 5, sizeof(double), &alpha);
	clSetKernelArg(kernel, 6, sizeof(double), &beta);
	clSetKernelArg(kernel, 7, 8, &x->real_size[1]);
	clSetKernelArg(kernel, 8, 8, &y->real_size[1]);
	clSetKernelArg(kernel, 9, 8, x->offset);
	clSetKernelArg(kernel, 10, 8, &x->offset[1]);
	clSetKernelArg(kernel, 11, 8, y->offset);
	clSetKernelArg(kernel, 12, 8, &y->offset[1]);

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, x->shape, &x->work_items[1], nw, be, e);
}

void wekuaBlasAsum(wmatrix *x, double *real, double *imag, uint32_t nw, cl_event *be){
	if (x == NULL){
		return;
	}
	wmatrix *a = wekuaMatrixAbs(x, nw, be);
	wekuaMatrixSum(a, real, imag, 0, NULL);
}

void wekuaBlasNorm(wmatrix *x, double *real, double *imag, uint32_t nw, cl_event *be){
	if (x == NULL){
		return;
	}
	cl_event e[2];

	wmatrix *b = wekuaMatrixCopy(x, nw, be, e);

	wekuaMatrixDot(b, x, 1, e, &e[1]);

	wekuaMatrixSum(b, real, imag, 1, &e[1]);

	clWaitForEvents(1, &e[1]);

	clReleaseEvent(e[0]);
	clReleaseEvent(e[1]);

	double ang, n;

	if (x->com){
		if (real[0] == 0){
			ang = CL_M_PI_2;
		}else{
			ang = atan(imag[0]/real[0])/2;
		}
		n = sqrt(real[0]*real[0]+imag[0]*imag[0]);
		real[0] = n*cos(ang);
		imag[0] = n*sin(ang);
	}else{
		real[0] = sqrt(real[0]);
	}
	wekuaFreeMatrix(b, 0, NULL);
}

/*************** GEMM **************/

void wekuaBlasGemm(double ralpha, double ialpha, uint8_t a_trans, wmatrix *a, uint8_t b_trans, wmatrix *b,
	double rbeta, double ibeta, wmatrix *c, uint32_t nw, cl_event *be, cl_event *e
){
	if (a == NULL || b == NULL || c == NULL){
		return;
	}

	if (a->com || b->com || c->com || ibeta != 0.0 || ialpha != 0.0){
		if (createComplexMatrix(a) || createComplexMatrix(b) || createComplexMatrix(c)){
			return;
		}
	}

	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[4];
	uint64_t *shapea = a->shape, *shapeb = b->real_size;
	uint64_t *offseta = a->offset, *offsetb = b->offset, *offsetc = c->offset;


	// Matrix
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &c->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &c->imag);

	// Transpose
	clSetKernelArg(kernel, 6, 1, &a_trans);
	clSetKernelArg(kernel, 7, 1, &b_trans);

	// Scalars
	clSetKernelArg(kernel, 8, sizeof(double), &ralpha);
	clSetKernelArg(kernel, 9, sizeof(double), &ialpha);
	clSetKernelArg(kernel, 10, sizeof(double), &rbeta);
	clSetKernelArg(kernel, 11, sizeof(double), &ibeta);

	// Dimensions
	clSetKernelArg(kernel, 12, 8, &shapea[1]);
	clSetKernelArg(kernel, 13, 8, &a->real_size[1]);
	clSetKernelArg(kernel, 14, 8, &shapeb[1]);
	clSetKernelArg(kernel, 15, 8, shapea);

	// Offsets
	clSetKernelArg(kernel, 16, 8, offseta);
	clSetKernelArg(kernel, 17, 8, &offseta[1]);
	clSetKernelArg(kernel, 18, 8, offsetb);
	clSetKernelArg(kernel, 19, 8, &offsetb[1]);
	clSetKernelArg(kernel, 20, 8, offsetc);
	clSetKernelArg(kernel, 21, 8, &offsetc[1]);

	// Does the matrix use complex numbers
	clSetKernelArg(kernel, 22, 1, &a->com);
	clSetKernelArg(kernel, 23, 8, &c->real_size[1]); // C Matrix dimension

	clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, c->shape, &c->work_items[1], nw, be, e);
}
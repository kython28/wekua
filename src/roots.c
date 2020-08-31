#include "wekua.h"
#include <math.h>

void runKernel(cl_command_queue cmd, cl_kernel kernel, uint32_t ndim, uint64_t *offsi, uint64_t *glosi, uint64_t *losi);

wmatrix *getUpperLowerBounds(wmatrix *a){
	wmatrix *degree, *b;
	double max;
	degree = wekuaMatrixResize(a, 1, a->shape[1]-1, 0.0, 0.0);

	b = wekuaAllocMatrix(a->ctx, 1, 2);
	wekuaMatrixAbs(degree);
	wekuaMatrixMax(degree, &max, NULL);

	b->raw_real[1] = 1.0 + max/fabs(a->raw_real[a->shape[1]-1]);
	
	wekuaFreeMatrix(degree);
	degree = wekuaCutMatrix(a, 1, a->shape[1]-1, 0, 1);
	wekuaMatrixAbs(degree);
	wekuaMatrixMax(degree, &max, NULL);

	b->raw_real[0] = fabs(a->raw_real[0])/(fabs(a->raw_real[0])+max);

	wekuaFreeMatrix(degree);

	return b;
}

wmatrix *getRoots(wmatrix *ran, uint32_t degree){
	wmatrix *roots, *radius, *angle;
	radius = wekuaMatrixRandUniform(ran->ctx, 1, degree, ran->raw_real[0], 0.0, ran->raw_real[1], 0.0, 0);
	angle = wekuaMatrixRandUniform(ran->ctx, 1, degree, 0.0, 0.0, CL_M_PI*2, 0.0, 0);
	roots = wekuaAllocComplexMatrix(ran->ctx, 1, degree);

	for (uint32_t x=0; x<degree; x++){
		roots->raw_real[x] = radius->raw_real[x]*cos(angle->raw_real[x]);
		roots->raw_imag[x] = radius->raw_real[x]*sin(angle->raw_real[x]);
	}
	wekuaFreeMatrix(radius);
	wekuaFreeMatrix(angle);
	return roots;
}

wmatrix *calc_dev(wmatrix *poly){
	wmatrix *a = wekuaAllocComplexMatrix(poly->ctx, poly->shape[0], poly->shape[1]-1);
	for (uint64_t x=1; x < poly->shape[1]; x++){
		a->raw_real[x-1] = x*poly->raw_real[x];
		a->raw_imag[x-1] = x*poly->raw_imag[x];
	}
	return a;
}


wmatrix *wekuaMatrixRoot(wmatrix *a){
	if (a == NULL){
		return NULL;
	}else if (a->shape[0] != 1){
		return NULL;
	}
	wekuaContext *ctx = a->ctx;
	cl_kernel kernel = ctx->kernels[21];

	wmatrix *ran, *roots, *d;
	ran = getUpperLowerBounds(a);
	roots = getRoots(ran, a->shape[1]-1);

	if (a->com == 0){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(roots);
			return NULL;
		}
	}
	d = calc_dev(a);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &roots->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &roots->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &d->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &d->imag);
	clSetKernelArg(kernel, 6, 8, &roots->shape[1]);

	runKernel(ctx->command_queue, kernel, 1, NULL, &roots->size, roots->work_items);

	wekuaFreeMatrix(d);
	wekuaFreeMatrix(ran);
	return roots;
}
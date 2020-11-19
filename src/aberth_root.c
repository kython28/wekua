#include "wekua.h"

wmatrix getUpperLowerBounds(wekuaContext ctx, wmatrix a, uint32_t dl, uint8_t dtype){
	void *max;
	wmatrix degree, b, c;
	uint64_t x, y;

	max = malloc(dl);

	b = wekuaMatrixResize(a, 1, a->shape[1]-1, NULL, NULL);
	degree = wekuaMatrixAbs(b, 0, NULL);
	wekuaFreeMatrix(b, 0, NULL);

	b = wekuaAllocMatrix(ctx, 1, 2, dtype);
	wekuaMatrixMax(degree, &y, &x, 0, NULL);
	wekuaGetValueFromMatrix(degree, y, x, max, NULL, 0, NULL);

	c = wekuaAllocMatrix(ctx, 1, a->shape[1]-1, dtype);

	if (dtype == WEKUA_DTYPE_DOUBLE){
		((double*)b->raw_real)[1] = 1.0 + ((double*)max)[0]/fabs(((double*)a->raw_real)[a->shape[1]-1]);
		memcpy(c->raw_real, &((double*)a->raw_real)[1], dl*c->shape[1]);
	}else{
		((float*)b->raw_real)[1] = 1.0 + ((float*)max)[0]/fabsf(((float*)a->raw_real)[a->shape[1]-1]);
		memcpy(c->raw_real, &((float*)a->raw_real)[1], dl*c->shape[1]);
	}

	wekuaFreeMatrix(degree, 0, NULL);
	degree = wekuaMatrixAbs(c, 0, NULL);
	wekuaFreeMatrix(c, 0, NULL);

	wekuaMatrixMax(degree, &y, &x, 0, NULL);
	wekuaGetValueFromMatrix(degree, y, x, max, NULL, 0, NULL);

	if (dtype == WEKUA_DTYPE_DOUBLE){
		((double*)b->raw_real)[0] = fabs(((double*)a->raw_real)[0])/(fabs(((double*)a->raw_real)[0]) + ((double*)max)[0]);
	}else{
		((double*)b->raw_real)[0] = fabsf(((float*)a->raw_real)[0])/(fabsf(((float*)a->raw_real)[0]) + ((float*)max)[0]);
	}
	
	wekuaFreeMatrix(degree, 0, NULL);
	free(max);

	return b;
}

wmatrix getRoots(wekuaContext ctx, wmatrix ran, uint64_t degree, uint8_t dtype){
	wmatrix roots, radius, angle;
	cl_event e;
	void *pi2 = malloc(ctx->dtype_length[dtype]);

	if (dtype == WEKUA_DTYPE_DOUBLE){
		((double*)pi2)[0] = CL_M_PI*2.0;
		radius = wekuaMatrixRandUniform(ctx, 1, degree, ran->raw_real, NULL, &((double*)ran->raw_real)[1], NULL, dtype);
	}else{
		((float*)pi2)[0] = CL_M_PI_F*2.0;
		radius = wekuaMatrixRandUniform(ctx, 1, degree, ran->raw_real, NULL, &((float*)ran->raw_real)[1], NULL, dtype);
	}
	if (radius == NULL){
		free(pi2);
		return NULL;
	}

	angle = wekuaMatrixRandUniform(ctx, 1, degree, NULL, NULL, pi2, NULL, dtype);
	if (angle == NULL){
		wekuaFreeMatrix(radius, 0, NULL);
		free(pi2);
		return NULL;
	}

	wekuaMatrixDot(angle, radius, 0, NULL, &e);

	roots = wekuaMatrixEulerIden(angle, 1, &e);

	clReleaseEvent(e);
	wekuaFreeMatrix(radius, 0, NULL);
	wekuaFreeMatrix(angle, 0, NULL);

	free(pi2);

	return roots;
}

wmatrix getDev(wekuaContext ctx, wmatrix poly, uint8_t dtype){
	if (compileKernel(ctx, WEKUA_KERNEL_ROOT_DEV, dtype)){
		return NULL;
	}

	cl_kernel kernel = ctx->kernels[WEKUA_KERNEL_ROOT_DEV*10+dtype];
	cl_event e;

	wmatrix dev = wekuaAllocComplexMatrix(ctx, 1, poly->shape[1]-1, poly->dtype);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &poly->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &poly->imag);
	clSetKernelArg(kernel, 4, 1, &poly->com);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &dev->shape[1], &dev->work_items[7], 0, NULL, &e);
	if (ret == CL_SUCCESS){
		clWaitForEvents(1, &e);
		clReleaseEvent(e);
	}else{
		wekuaFreeMatrix(dev, 0, NULL);
	}
	
	return dev;
}

wmatrix wekuaMatrixRoot(wmatrix poly, uint32_t nw, cl_event *be){
	if (poly == NULL) return NULL;

	clWaitForEvents(nw, be);

	wekuaContext ctx = poly->ctx;

	uint8_t dtype = poly->dtype;
	uint32_t dl = ctx->dtype_length[dtype];
	wmatrix ran, roots, dev;
	cl_kernel kernel;
	cl_event e;
	
	if (compileKernel(ctx, WEKUA_KERNEL_ROOT, dtype)) return NULL;

	if (createComplexMatrix(poly)) return NULL;

	kernel = ctx->kernels[WEKUA_KERNEL_ROOT*10+dtype];

	ran = getUpperLowerBounds(ctx, poly, dl, dtype);
	roots = getRoots(ctx, ran, poly->shape[1]-1, dtype);
	dev = getDev(ctx, poly, dtype);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &roots->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &roots->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &poly->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &poly->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &dev->imag);
	clSetKernelArg(kernel, 6, 8, &roots->shape[1]);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &roots->shape[1], &roots->work_items[7], 0, NULL, &e);
	if (ret == CL_SUCCESS){
		wekuaFreeMatrix(dev, 1, &e);
		clReleaseEvent(e);
	}else{
		wekuaFreeMatrix(dev, 0, NULL);
		wekuaFreeMatrix(roots, 0, NULL);
		roots = NULL;
	}

	return roots;
}
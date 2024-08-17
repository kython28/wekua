#include "../headers/matrix.h"
#include <math.h>

wmatrix getUpperLowerBounds(wekuaContext ctx, wmatrix a, uint32_t dl, uint8_t dtype){
	wmatrix degree, b, c;
	uint64_t x, y;

	cl_event event;
	cl_command_queue cmd = ctx->command_queue;

	char max[sizeof(double)], a_scal[sizeof(double)], b_scal[sizeof(double)];

	b = wekuaMatrixResize(a, 1, a->shape[1]-1, NULL, NULL);
	degree = wekuaMatrixAbs(b, 0, NULL);
	wekuaFreeMatrix(b, 0, NULL);

	b = wekuaAllocMatrix(ctx, 1, 2, dtype);
	wekuaMatrixMax(degree, &y, &x, 0, NULL);
	wekuaGetValueFromMatrix(degree, y, x, max, NULL, 0, NULL);

	c = wekuaAllocMatrix(ctx, 1, a->shape[1]-1, dtype);

	wekuaGetValueFromMatrix(a, 0, a->shape[1]-1, a_scal, NULL, 0, NULL);
	wekuaGetValueFromMatrix(b, 0, 1, b_scal, NULL, 0, NULL);
	if (dtype == WEKUA_DTYPE_DOUBLE){		
		((double*)b_scal)[0] = ((double*)max)[0]/fabs(((double*)a_scal)[0]);

	}else{
		((float*)b_scal)[0] = ((float*)max)[0]/fabsf(((float*)a_scal)[0]);
	}

	wekuaPutValueToMatrix(b, 0, 1, b_scal, NULL, 0, NULL);
	clEnqueueCopyBuffer(cmd, a->real, c->real, dl, 0, dl*c->shape[1], 0, NULL, &event);
	clWaitForEvents(1, &event);
	clReleaseEvent(event);

	wekuaFreeMatrix(degree, 0, NULL);
	degree = wekuaMatrixAbs(c, 0, NULL);
	wekuaFreeMatrix(c, 0, NULL);

	wekuaMatrixMax(degree, &y, &x, 0, NULL);

	wekuaGetValueFromMatrix(degree, y, x, max, NULL, 0, NULL);
	wekuaGetValueFromMatrix(a, 0, 0, a_scal, NULL, 0, NULL);
	wekuaGetValueFromMatrix(b, 0, 0, b_scal, NULL, 0, NULL);

	if (dtype == WEKUA_DTYPE_DOUBLE){
		((double*)b_scal)[0] = fabs(((double*)a_scal)[0])/(fabs(((double*)a_scal)[0]) + ((double*)max)[0]);
	}else{
		((float*)b_scal)[0] = fabsf(((float*)a_scal)[0])/(fabsf(((float*)a_scal)[0]) + ((float*)max)[0]);
	}
	wekuaPutValueToMatrix(b, 0, 0, b_scal, NULL, 0, NULL);

	wekuaFreeMatrix(degree, 0, NULL);

	return b;
}

wmatrix getRoots(wekuaContext ctx, wmatrix ran, uint64_t degree, uint8_t dtype){
	wmatrix roots, radius, angle;
	cl_event e;

	char scal1[sizeof(double)], scal2[sizeof(double)], pi2[sizeof(double)];

	wekuaGetValueFromMatrix(ran, 0, 0, scal1, NULL, 0, NULL);
	wekuaGetValueFromMatrix(ran, 0, 0, scal2, NULL, 0, NULL);

	
	if (dtype == WEKUA_DTYPE_DOUBLE) ((double*)pi2)[0] = CL_M_PI*2.0;
	else ((float*)pi2)[0] = CL_M_PI_F*2.0f;

	radius = wekuaMatrixRandUniform(ctx, 1, degree, scal1, NULL, scal2, NULL, dtype);
	if (radius == NULL) return NULL;

	angle = wekuaMatrixRandUniform(ctx, 1, degree, NULL, NULL, pi2, NULL, dtype);
	if (angle == NULL){
		wekuaFreeMatrix(radius, 0, NULL);
		return NULL;
	}

	int ret = wekuaMatrixDot(angle, radius, 0, NULL, &e);
	if (ret != CL_SUCCESS) {
		wekuaFreeMatrix(angle, 0, NULL);
		wekuaFreeMatrix(radius, 0, NULL);
		return NULL;
	}

	roots = wekuaMatrixEulerIden(angle, 1, &e);

	clReleaseEvent(e);
	wekuaFreeMatrix(radius, 0, NULL);
	wekuaFreeMatrix(angle, 0, NULL);

	return roots;
}

wmatrix getDev(wekuaContext ctx, wmatrix poly, uint8_t dtype){
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_ROOT_DEV, dtype, poly->com);
	if (kernel == NULL) return NULL;

	cl_event e;
	wmatrix dev = wekuaAllocComplexMatrix(ctx, 1, poly->shape[1]-1, poly->dtype);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &poly->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &poly->imag);

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
	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_ROOT, dtype, 1);
	cl_event e;
	
	if (kernel == NULL) return NULL;

	if (createComplexMatrix(poly)) return NULL;

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

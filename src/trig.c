#include "wekua.h"

int wTrig(wmatrix a, uint8_t kn, uint32_t nw, cl_event *be, cl_event *e){
	if (a->dtype < WEKUA_DTYPE_FLOAT){
		return CL_INVALID_MEM_OBJECT;
	}

	wekuaContext ctx = a->ctx;
	if (compileKernel(ctx, kn, a->dtype)){
		return CL_COMPILE_PROGRAM_FAILURE;
	}

	cl_kernel kernel = ctx->kernels[kn*10+a->dtype];

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(kernel, 2, 8, &a->vl_shape[1]);
	clSetKernelArg(kernel, 3, 1, &a->com);

	return clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, a->vl_shape, a->work_items, nw, be, e);
}

int wekuaMatrixSin(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	return wTrig(a, WEKUA_KERNEL_SIN, nw, be, e);
}

int wekuaMatrixCos(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	return wTrig(a, WEKUA_KERNEL_COS, nw, be, e);
}

int wekuaMatrixTan(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	return wTrig(a, WEKUA_KERNEL_TAN, nw, be, e);
}

int wekuaMatrixSinh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	return wTrig(a, WEKUA_KERNEL_SINH, nw, be, e);
}

int wekuaMatrixCosh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	return wTrig(a, WEKUA_KERNEL_COSH, nw, be, e);
}

int wekuaMatrixTanh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e){
	if (a == NULL){
		return CL_INVALID_MEM_OBJECT;
	}
	return wTrig(a, WEKUA_KERNEL_TANH, nw, be, e);
}
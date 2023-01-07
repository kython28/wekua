#include "../headers/error.h"

int runLossKernel(wmatrix output, wmatrix output_wanted, void *error_scal, void *errori_scal, werror *err, uint32_t nw, cl_event *be, uint8_t kernel_id){
	if (output == NULL || output_wanted == NULL) return CL_INVALID_MEM_OBJECT;

	int ret;
	cl_event e;
	wekuaContext ctx = output->ctx;
	cl_kernel kernel;
	uint8_t dtype = output->dtype, dev = 0, com = output->com|output_wanted->com;
	wmatrix error;
	
	wmatrix dev_m = NULL;
	cl_mem *dev_r = NULL, *dev_i = NULL;

	if (com){
		if (createComplexMatrix(output_wanted)|createComplexMatrix(output)){
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}
	}

	if (err != NULL){
		dev = 1;
		err[0] = (werror) calloc(1, sizeof(struct _w_error));
		if (com){
			dev_m = wekuaAllocComplexMatrix(ctx, output->shape[0], output->shape[1], dtype);
		}else{
			dev_m = wekuaAllocMatrix(ctx, output->shape[0], output->shape[1], dtype);
		}
		if (dev_m == NULL){
			free(err[0]);
			return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}

		err[0]->err = dev_m;
		
		dev_r = &dev_m->real;
		dev_i = &dev_m->imag;
	}

	kernel = compileKernel(ctx, kernel_id, dtype, com);
	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	if (com){
		error = wekuaAllocComplexMatrix(ctx, output->shape[0], output->shape[1], dtype);
	}else{
		error = wekuaAllocMatrix(ctx, output->shape[0], output->shape[1], dtype);
	}
	if (error == NULL) return CL_MEM_OBJECT_ALLOCATION_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_wanted->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_wanted->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &output->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &output->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &error->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &error->imag);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), dev_r);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), dev_i);
	clSetKernelArg(kernel, 8, 8, &output->vl_shape[1]);
	clSetKernelArg(kernel, 9, 1, &dev);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, output->vl_shape, output->work_items, nw, be, &e);
	if (ret != CL_SUCCESS) goto wekua_mse_fail;

	if (error_scal || errori_scal){
		ret = wekuaMatrixMean(error, error_scal, errori_scal, 1, &e);
		if (ret != CL_SUCCESS) clWaitForEvents(1, &e);
	}else{
		clWaitForEvents(1, &e);
	}
	
	clReleaseEvent(e);

	wekua_mse_fail:
	if (ret != CL_SUCCESS){
		if (err != NULL){
			wekuaFreeMatrix(dev_m, 0, NULL);	
			free(err[0]);
		}
	}
	wekuaFreeMatrix(error, 0, NULL);

	return ret;
}

int wekuaMSE(wmatrix output, wmatrix output_wanted, void *error_scal, void *errori_scal, werror *err, uint32_t nw, cl_event *be){
	return runLossKernel(output, output_wanted, error_scal, errori_scal, err, nw, be, WEKUA_KERNEL_MSE);
}
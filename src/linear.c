#include "wekua.h"

void *get_one(uint8_t dtype, uint32_t dl);

wmatrix run_linear(void *m, wmatrix input, wcache *cache, uint32_t nw, cl_event *be);
int run_linear_bias(cl_kernel kernel, cl_command_queue cmd, wmatrix output, wmatrix bias, uint32_t dl, cl_event *e);


wneuron wekuaLinear(wekuaContext ctx, uint64_t input, uint64_t output, uint64_t deep, uint8_t bias, wacti acti, uint8_t dtype){
	if (input == 0 || output == 0 || deep == 0 || acti == NULL) return NULL;
	else if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	if (bias){
		if (compileKernel(ctx, WEKUA_KERNEL_BIAS, dtype)) return NULL;
	}

	if (compileKernel(ctx, WEKUA_KERNEL_GEMM, dtype)) return NULL;

	wmatrix tmp;
	cl_event e;

	wneuron neur = (wneuron) calloc(1, sizeof(struct _w_neuron));
	if (neur == NULL) return NULL;

	neur->layer = deep;
	neur->dtype = dtype;
	neur->acti = acti;

	neur->weight = (wmatrix*) calloc(deep, sizeof(wmatrix));
	if (neur->weight == NULL) return NULL;

	neur->weight[0] = wekuaMatrixRandn(ctx, output, input, 0);
	if (neur->weight[0] == NULL) goto wekua_linear_fail;

	for (uint64_t x=1; x<deep; x++){
		neur->weight[x] = wekuaMatrixRandn(ctx, output, output, 0);
		if (neur->weight[x] == NULL) goto wekua_linear_fail;
	}

	if (dtype != WEKUA_DTYPE_DOUBLE){
		for (uint64_t x=0; x<deep; x++){
			tmp = neur->weight[x];
			neur->weight[x] = wekuaMatrixConvert(tmp, dtype, 0, NULL, &e);
			if (neur->weight[x] == NULL){
				neur->weight[x] = tmp;
				goto wekua_linear_fail;
			}

			wekuaFreeMatrix(tmp, 1, &e);
			clReleaseEvent(e);
		}
	}

	if (bias){
		neur->bias = (wmatrix*) calloc(deep, sizeof(wmatrix));
		if (neur->bias == NULL) goto wekua_linear_fail;

		for (uint64_t x=0; x<deep; x++){
			neur->bias[x] = wekuaMatrixRandn(ctx, 1, output, 0);
			if (neur->bias[x] == NULL) goto wekua_linear_fail;
		}

		if (dtype != WEKUA_DTYPE_DOUBLE){
			for (uint64_t x=0; x<deep; x++){
				tmp = neur->bias[x];
				neur->bias[x] = wekuaMatrixConvert(tmp, dtype, 0, NULL, &e);
				if (neur->weight[x] == NULL){
					neur->bias[x] = tmp;
					goto wekua_linear_fail;
				}

				wekuaFreeMatrix(tmp, 1, &e);
				clReleaseEvent(e);
			}
		}
	}

	neur->run = &run_linear;

	goto wekua_linear_success;

	wekua_linear_fail:
	if (neur->weight != NULL){
		for (uint64_t x=0; x<deep; x++){
			wekuaFreeMatrix(neur->weight[x], 0, NULL);
		}
		free(neur->weight);
	}

	if (neur->bias != NULL){
		for (uint64_t x=0; x<deep; x++){
			wekuaFreeMatrix(neur->bias[x], 0, NULL);
		}
		free(neur->bias);
	}

	free(neur);
	neur = NULL;

	wekua_linear_success:
	return neur;
}

wmatrix run_linear(void *m, wmatrix input, wcache *cache, uint32_t nw, cl_event *be){
	if (m == NULL || input == NULL) return NULL;
	else if (input->dtype < WEKUA_DTYPE_FLOAT) return NULL;

	wekuaContext ctx = input->ctx;
	wneuron linear = m;

	wmatrix *weight = linear->weight;
	wmatrix *bias = linear->bias;
	wmatrix *data_cache;
	wacti acti = linear->acti;
	int ret;
	uint8_t dtype = linear->dtype;
	uint64_t layer = linear->layer;
	uint32_t dl = ctx->dtype_length[dtype];
	void *one;

	cl_event e;
	cl_kernel kernel;
	cl_command_queue cmd = ctx->command_queue;

	kernel = ctx->kernels[WEKUA_KERNEL_BIAS*10+dtype];

	if (cache != NULL){
		cache[0] = (wcache) calloc(1, sizeof(struct _w_cache));
		if (cache[0] == NULL) return NULL;

		cache[0]->ndata = layer+1;
		data_cache = (wmatrix*) calloc(layer+1, sizeof(wmatrix));
		if (data_cache == NULL){
			free(cache[0]);
			return NULL;
		}
		cache[0]->data = data_cache;
		data_cache[0] = input;
	}

	one = get_one(dtype, dl);
	if (one == NULL) goto wekua_rli_fail;

	clWaitForEvents(nw, be);

	wmatrix output = NULL, in = input;
	for (uint64_t x=0; x<layer; x++){ 
		output = wekuaAllocMatrix(ctx, in->shape[0], weight[x]->shape[0], dtype);
		if (output == NULL) goto wekua_rli_fail;

		ret = wekuaBlasGemm(one, NULL, 0, in, 1, weight[x], NULL, NULL, output, 0, NULL);
		if (ret != CL_SUCCESS) goto wekua_rli_fail;

		if (bias != NULL){
			ret = run_linear_bias(kernel, cmd, output, bias[x], dl, &e);
			if (ret != CL_SUCCESS) goto wekua_rli_fail;

			clWaitForEvents(1, &e);
			clReleaseEvent(e);
		}

		runWekuaActi(acti, output, 0, NULL);

		if (cache == NULL){
			if (in != input) wekuaFreeMatrix(in, 0, NULL);
		}else{
			cache[0]->data[x+1] = output;
		}

		in = output;
	}

	goto wekua_rli_success;

	wekua_rli_fail:
	wekuaFreeMatrix(output, 0, NULL);
	output = NULL;

	if (in != input) wekuaFreeMatrix(in, 0, NULL);

	if (cache != NULL){
		if (cache[0] != NULL){
			for (uint64_t x=0; x<=layer; x++){
				wekuaFreeMatrix(data_cache[x], 0, NULL);
			}
			free(data_cache);
			free(cache[0]);
		}
	}

	wekua_rli_success:
	if (one != NULL) free(one);

	return output;
}

int run_linear_bias(cl_kernel kernel, cl_command_queue cmd, wmatrix output, wmatrix bias, uint32_t dl, cl_event *e){
	if (output->com){
		if (createComplexMatrix(bias)){
			return 1;
		}
	}

	uint64_t local_si = sizeof(cl_mem)*dl*output->work_items[1];
	
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &bias->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bias->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &output->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &output->imag);

	clSetKernelArg(kernel, 4, 8, &output->vl_shape[1]);
	clSetKernelArg(kernel, 5, 1, &output->com);

	clSetKernelArg(kernel, 6, local_si, NULL);
	clSetKernelArg(kernel, 7, local_si, NULL);

	return clEnqueueNDRangeKernel(cmd, kernel, 2, NULL, output->vl_shape, output->work_items, 0, NULL, e);
}
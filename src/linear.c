#include "wekua.h"

void getLWI(uint64_t *x, uint64_t *y, uint32_t si, uint64_t max);
void *get_one(uint8_t dtype, uint32_t dl);

wmatrix run_linear(void *m, wmatrix input, wcache *cache, uint32_t nw, cl_event *be);
int run_linear_bias(cl_kernel kernel, cl_command_queue cmd, wmatrix output, wmatrix bias, uint32_t dl, cl_event *e);
int backward_linear(void *n, werror error, wcache cache, werror *err);
int step_linear(void *neur, void *opti_data, void *other, werror error, wcache cache, int (*func)(void *, void *, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix));
wmatrix get_dev_bias(wekuaContext ctx, wmatrix error, uint8_t dtype);

void free_error_linear(werror err);
void free_cache_linear(wcache cache);

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
	neur->step = &step_linear;
	neur->free_cache = &free_cache_linear;
	neur->free_error = &free_error_linear;
	neur->run = &run_linear;
	neur->backward = &backward_linear;

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
			((wmatrix*)cache[0]->data)[x+1] = output;
		}

		in = output;
	}

	goto wekua_rli_success;

	wekua_rli_fail:
	wekuaFreeMatrix(output, 0, NULL);
	output = NULL;

	if (cache != NULL){
		if (cache[0] != NULL){
			for (uint64_t x=0; x<=layer; x++){
				wekuaFreeMatrix(data_cache[x], 0, NULL);
			}
			free(data_cache);
			free(cache[0]);
		}
	}else if (in != input) wekuaFreeMatrix(in, 0, NULL);

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

int backward_linear(void *n, werror error, wcache cache, werror *err){
	if (n == NULL || error == NULL || cache == NULL) return CL_INVALID_ARG_VALUE;

	int ret;

	wekuaContext ctx;
	wneuron neuron = n;
	
	uint8_t dtype = neuron->dtype;

	wacti acti = neuron->acti;
	cl_event e;
	
	wmatrix *cache_data = cache->data;
	wmatrix dev, *o_err, *weight, tmp;

	weight = neuron->weight;
	ctx = weight[0]->ctx;

	void *one = get_one(dtype, ctx->dtype_length[dtype]);
	uint64_t no_err = neuron->layer;
	
	error->no_err = no_err;
	o_err = (wmatrix*) calloc(no_err, sizeof(wmatrix));
	error->o_err = o_err;
	
	tmp = error->err;
	wmatrix s, w;

	for (; no_err>0; no_err--){
		o_err[no_err-1] = tmp;

		dev = wekuaActiGetDev(acti, cache_data[no_err]);
		ret = wekuaMatrixDot(tmp, dev, 0, NULL, &e);
		wekuaFreeMatrix(dev, 0, NULL);
		if (ret != CL_SUCCESS) break;

		s = tmp;
		w = weight[no_err-1];

		tmp = wekuaAllocMatrix(ctx, s->shape[0], w->shape[1], dtype);
		ret = wekuaBlasGemm(one, NULL, 0, s, 0, w, NULL, NULL, tmp, 1, &e);
		if (ret != CL_SUCCESS){
			clWaitForEvents(1, &e);
			clReleaseEvent(e);
			break;
		}

		clReleaseEvent(e);
		
	}
	free(one);

	if (ret == CL_SUCCESS && err != NULL){
		err[0] = (werror) calloc(1, sizeof(struct _w_error));
		err[0]->err = tmp;
	}else{
		wekuaFreeMatrix(tmp, 0, NULL);
	}

	return ret;
}

wmatrix get_dev_bias(wekuaContext ctx, wmatrix error, uint8_t dtype){
	cl_event e;
	uint8_t com = error->com;
	cl_kernel kernel;
	uint64_t row, wi;

	row = error->shape[0];

	if (compileKernel(ctx, WEKUA_KERNEL_LINEAR_BIAS_STEP, dtype)){
		return NULL;
	}

	kernel = ctx->kernels[10*WEKUA_KERNEL_LINEAR_BIAS_STEP+dtype];

	wmatrix dev = wekuaAllocMatrix(ctx, 1, row, dtype);
	if (com){
		if (createComplexMatrix(dev)){
			wekuaFreeMatrix(dev, 0, NULL);
			return NULL;
		}
	}

	if (row%2 != 0) row++;
	row /= 2;
	getLWI(&row, &wi, 1, ctx->max_work_group_size);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &error->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &error->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev->imag);
	clSetKernelArg(kernel, 4, 8, &error->vl_shape[1]);
	clSetKernelArg(kernel, 5, 1, &com);

	int ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 1, NULL, &row, &wi,
		0, NULL, &e
	);

	if (ret != CL_SUCCESS){
		wekuaFreeMatrix(dev, 0, NULL);
		return NULL;
	}

	clWaitForEvents(1, &e);
	clReleaseEvent(e);

	return dev;
}

int step_linear(void *neur, void *opti_data, void *other, werror error, wcache cache, int (*func)(void *, void *, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix)){
	wneuron neuron = neur;
	uint64_t layers = neuron->layer, x = 0;
	uint8_t dtype = neuron->dtype;
	uint32_t dl;
	cl_event event;
	int ret;
	
	wekuaContext ctx;
	wmatrix *weigth = neuron->weight;
	wmatrix *bias = neuron->bias;
	wmatrix *cache_d = cache->data;
	wmatrix *errors = error->o_err;
	wmatrix **grad = other;

	wmatrix dev, dev_bias = NULL;
	ctx = weigth[0]->ctx;
	dl = ctx->dtype_length[dtype];

	void *one = get_one(dtype, dl);
	wmatrix w, e, a, *g = NULL;
	for (; x<layers; x++){
		w = weigth[x];
		e = errors[x];
		a = cache_d[x];

		if (grad != NULL) g = grad[x];

		e = wekuaMatrixTrans(errors[x], 0, NULL, &event);

		dev = wekuaAllocMatrix(ctx, w->shape[0], w->shape[1], dtype);
		ret = wekuaBlasGemm(one, NULL, 0, e, 0, a, NULL, NULL, dev, 1, &event);
		clReleaseEvent(event);

		if (bias != NULL){
			dev_bias = get_dev_bias(ctx, e, dtype);
			if (ret != CL_SUCCESS || dev_bias == NULL){
				wekuaFreeMatrix(dev, 0, NULL);
				break;
			}

			ret = func(opti_data, g, dl, dev, dev_bias, w, bias[x]);
			wekuaFreeMatrix(dev_bias, 0, NULL);
		}else{
			ret = func(opti_data, g, dl, dev, NULL, w, NULL);
		}

		wekuaFreeMatrix(dev, 0, NULL);
		wekuaFreeMatrix(e, 0, NULL);
		if (ret != CL_SUCCESS) break;
	}

	free(one);
	return ret;
}

void free_error_linear(werror err){
	uint64_t no_err = err->no_err;
	wmatrix *o_errors = err->o_err;
	for (uint64_t x=0; x<no_err; x++) wekuaFreeMatrix(o_errors[x], 0, NULL);

	free(o_errors);
	free(err);
}

void free_cache_linear(wcache cache){
	uint64_t ndata = cache->ndata;
	wmatrix *data = cache->data;
	for (uint64_t x=1; x<ndata; x++) wekuaFreeMatrix(data[x], 0, NULL);

	free(data);
	free(cache);
}
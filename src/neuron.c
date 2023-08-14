#include "../headers/neuron.h"
#include "regularization.h"

// ------------------- HELPERS ------------------- //
void getLWI(uint64_t *x, uint64_t *y, uint32_t si, uint64_t max);
void *get_one(uint8_t dtype, uint32_t dl);

static int run_neuron_bias(wekuaContext ctx, cl_command_queue cmd, cl_device_local_mem_type local_mem_type, wmatrix output, wmatrix bias, uint32_t dl, cl_event *e){
	uint8_t com = output->com|bias->com;
	if (output->com){
		if (createComplexMatrix(bias)){
			return 1;
		}
	}

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_BIAS, bias->dtype, com);
	uint64_t local_si;
	
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &bias->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bias->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &output->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &output->imag);
	clSetKernelArg(kernel, 4, 8, &output->vl_shape[1]);

	if (local_mem_type == CL_LOCAL){
		local_si = sizeof(cl_mem)*dl*output->work_items[1];

		clSetKernelArg(kernel, 5, local_si, NULL);
		clSetKernelArg(kernel, 6, local_si, NULL);
	}

	return clEnqueueNDRangeKernel(cmd, kernel, 2, NULL, output->vl_shape, output->work_items, 0, NULL, e);
}

// ----------------------------------------------- //

wmatrix runWekuaNeuron(wneuron neuron, wmatrix input, wcache *cache, uint32_t nw, cl_event *be){
	if (neuron == NULL || input == NULL) return NULL;
	else if (input->dtype < WEKUA_DTYPE_FLOAT) return NULL;

	wekuaContext ctx = input->ctx;

	wmatrix *weight = neuron->weight;
	wmatrix *bias = neuron->bias;
	wmatrix *data_cache = NULL;
	wacti acti = neuron->acti;
	int ret;
	uint8_t dtype = neuron->dtype;
	uint64_t layer = neuron->layer;
	uint32_t dl = ctx->dtype_length[dtype];
	cl_device_local_mem_type local_mem_type = ctx->local_mem_type;
	void *one;

	cl_event e;
	cl_command_queue cmd = ctx->command_queue;

	if (cache){
		if (!cache[0]){
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
		}else{
			data_cache = cache[0]->data;
			if (data_cache[0]->shape[0] != input->shape[0]) return NULL;

			data_cache[0] = input;
		}
	}

	wmatrix output = NULL, in = input;

	one = get_one(dtype, dl);
	if (one == NULL) goto wekua_rli_fail;

	clWaitForEvents(nw, be);
	for (uint64_t x=0; x<layer; x++){
		if (cache){
			output = data_cache[x+1];
			if (!output){
				output = wekuaAllocMatrix(ctx, in->shape[0], weight[x]->shape[0], dtype);
				if (output == NULL) goto wekua_rli_fail;

				data_cache[x+1] = output;
			}
		}

		ret = wekuaBlasGemm(one, NULL, 0, in, 1, weight[x], NULL, NULL, output, 0, NULL);
		if (ret != CL_SUCCESS) goto wekua_rli_fail;

		if (bias){
			ret = run_neuron_bias(ctx, cmd, local_mem_type, output, bias[x], dl, &e);
			if (ret != CL_SUCCESS) goto wekua_rli_fail;

			clWaitForEvents(1, &e);
			clReleaseEvent(e);
		}

		runWekuaActi(acti, output, 0, NULL);

		if (!cache && in != input) wekuaFreeMatrix(in, 0, NULL);
		in = output;
	}

	goto wekua_rli_success;

	wekua_rli_fail:
	wekuaFreeMatrix(output, 0, NULL);
	output = NULL;

	if (in != input) wekuaFreeMatrix(in, 0, NULL);

	wekua_rli_success:
	if (one) free(one);

	return output;
}

int wekuaNeuronBackward(wneuron neuron, werror error, wcache cache, wmatrix regularization, werror *err){
	if (neuron == NULL || error == NULL || cache == NULL) return CL_INVALID_ARG_VALUE;

	int ret;

	wekuaContext ctx;
	
	uint8_t dtype = neuron->dtype;

	wacti acti = neuron->acti;
	cl_event e[2];
	uint32_t nevents = 0;
	
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
		if (dev == NULL) break;

		ret = wekuaMatrixDot(tmp, dev, 0, NULL, e);
		if (ret != CL_SUCCESS) break;
		nevents++;

		if (regularization){
			ret = wekuaAddRegularization(regularization, tmp, 1, e, &e[1]);
			if (ret != CL_SUCCESS) break;

			regularization = NULL;
			nevents++;
		}

		s = tmp;
		w = weight[no_err-1];

		tmp = wekuaAllocMatrix(ctx, s->shape[0], w->shape[1], dtype);
		ret = wekuaBlasGemm(one, NULL, 0, s, 0, w, NULL, NULL, tmp, 1, &e[nevents - 1]);
		if (ret != CL_SUCCESS){
			wekuaFreeMatrix(dev, 1, &e[nevents - 1]);
			break;
		}
		clReleaseEvent(e[0]);
		if (nevents == 2) clReleaseEvent(e[1]);
		wekuaFreeMatrix(dev, 0, NULL);

		nevents = 0;
	}
	free(one);

	if (ret == CL_SUCCESS && err != NULL){
		err[0] = (werror) calloc(1, sizeof(struct _w_error));
		err[0]->err = tmp;
	}else{
		wekuaFreeMatrix(tmp, 0, NULL);

		if (nevents){
			clWaitForEvents(nevents, e);
			for (uint32_t i=0; i<nevents; i++) clReleaseEvent(e[i]);
		}
	}

	return ret;
}

void wekuaFreeNeuronError(werror err){
	uint64_t no_err = err->no_err;
	wmatrix *o_errors = err->o_err;
	for (uint64_t x=0; x<no_err; x++) wekuaFreeMatrix(o_errors[x], 0, NULL);

	free(o_errors);
	free(err);
}

void wekuaFreeNeuronCache(wcache cache){
	uint64_t ndata = cache->ndata;
	wmatrix *data = cache->data;
	for (uint64_t x=1; x<ndata; x++) wekuaFreeMatrix(data[x], 0, NULL);

	free(data);
	free(cache);
}

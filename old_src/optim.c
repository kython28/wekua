#include "../headers/optim.h"
#include <assert.h>

void *get_one(uint8_t dtype, uint32_t dl);

void exchange_ptr(wmatrix a, wmatrix b){
	void *tmp;
	tmp = a->real;
	a->real = b->real;
	b->real = tmp;

	tmp = a->imag;
	a->imag = b->imag;
	b->imag = tmp;

	// tmp = a->raw_real;
	// a->raw_real = b->raw_real;
	// b->raw_real = tmp;

	// tmp = a->raw_imag;
	// a->raw_imag = b->raw_imag;
	// b->raw_imag = tmp;
}

void zero_optim(wmatrix a){
	if (a == NULL) return;
	mem_set_zero(a, a->real);
	if (a->com) mem_set_zero(a, a->imag);
}

// Gradient Descent

int step_optim_gd(void *data, __attribute__((unused)) void *grad, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weight, wmatrix bias){
	void *lr = data;
	void *lri = data + dl;
	uint32_t evn = 0;
	cl_event e[2];
	int ret;

	ret = wekuaBlasAxpy(error, weight, lr, lri, 0, NULL, e);
	if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;
	evn++;

	if (error_b != NULL){
		ret = wekuaBlasAxpy(error_b, bias, lr, lri, 0, NULL, &e[1]);
		if (ret != CL_SUCCESS) goto wk_step_optim_gd_fail;
		evn++;
	}

	wk_step_optim_gd_fail:
	clWaitForEvents(evn, e);
	for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);
	return ret;
}

void free_optim_gd(void *opti, uint32_t nw, cl_event *be){
	if (opti == NULL) return;
	clWaitForEvents(nw, be);
	woptim optim = opti;

	free(optim->params);
	free(opti);
}

woptim wekuaOptimGD(wekuaContext ctx, wnetwork net, void *lr, void *lri, uint8_t dtype){
	if ((lr == NULL && lri == NULL) || net == NULL || ctx == NULL) return NULL;
	else if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	uint32_t dl = ctx->dtype_length[dtype];
	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	if (opti == NULL) return NULL;

	void *data = calloc(2, dl);
	if (data == NULL) {
		free(opti);
		return NULL;
	}

	if (dtype == WEKUA_DTYPE_FLOAT){
		if (lr != NULL) ((float*)data)[0] = -1.0f*((float*)lr)[0];
		if (lri != NULL) ((float*)data)[1] = -1.0f*((float*)lri)[0];
	}else{
		if (lr != NULL) ((double*)data)[0] = -1.0*((double*)lr)[0];
		if (lri != NULL) ((double*)data)[1] = -1.0*((double*)lri)[0];
	}
	
	opti->net = net;
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->params = data;
	opti->step = &step_optim_gd;
	opti->free_func = &free_optim_gd;

	return opti;
}

// Gradient Descent momentum

int step_optim_gdm(void *data, void *grad, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weight, wmatrix bias){
	void *lr = data;
	void *lri = data + dl;
	void *betar = data + (dl << 1);
	void *betai = data + dl*3;

	int ret;
	wekuaContext ctx = error->ctx;
	uint8_t dtype = error->dtype, com = weight->com;
	uint32_t evn = 0;
	cl_event e[2];

	wmatrix *gradient = grad;

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_GDM, dtype, com);
	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &error->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &error->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &weight->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &gradient[0]->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &gradient[0]->imag);
	clSetKernelArg(kernel, 6, dl, lr);
	clSetKernelArg(kernel, 7, dl, lri);
	clSetKernelArg(kernel, 8, dl, betar);
	clSetKernelArg(kernel, 9, dl, betai);
	clSetKernelArg(kernel, 10, 8, &weight->vl_shape[1]);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, weight->vl_shape, weight->work_items, 0, NULL, e);
	if (ret != CL_SUCCESS) goto wk_step_optim_gdm_fail;
	evn++;

	if (error_b != NULL){
		if (com){
			if (createComplexMatrix(bias)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &error_b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &error_b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias->real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias->imag);
		clSetKernelArg(kernel, 4, sizeof(cl_mem), &gradient[1]->real);
		clSetKernelArg(kernel, 5, sizeof(cl_mem), &gradient[1]->imag);
		clSetKernelArg(kernel, 6, dl, lr);
		clSetKernelArg(kernel, 7, dl, lri);
		clSetKernelArg(kernel, 8, dl, betar);
		clSetKernelArg(kernel, 9, dl, betai);
		clSetKernelArg(kernel, 10, 8, &bias->vl_shape[1]);

		ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, bias->vl_shape, bias->work_items, 0, NULL, &e[1]);
		if (ret != CL_SUCCESS) goto wk_step_optim_gdm_fail;
		evn++;
	}

	wk_step_optim_gdm_fail:
	clWaitForEvents(evn, e);
	for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);

	return ret;
}

int zero_optim_gdm(void *opti){
	woptim optim = opti;
	wnetwork net = optim->net;
	wneuron neuron_tmp;
	wmatrix ***velocity = optim->others;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			zero_optim(velocity[i][j][0]);
			zero_optim(velocity[i][j][1]);
		}
	}
	return CL_SUCCESS;
}

void free_optim_gdm(void *opti, uint32_t nw, cl_event *be){
	if (opti == NULL) return;
	clWaitForEvents(nw, be);
	woptim optim = opti;

	wnetwork net = optim->net;
	wmatrix ***velocity = optim->others;
	wneuron neuron_tmp;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	free(optim->params);

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
			free(velocity[i][j]);
		}
		free(velocity[i]);
	}
	free(velocity);
	free(opti);
}

woptim wekuaOptimGDM(wekuaContext ctx, wnetwork net, void *lr, void *lri, void *beta, void *betai, uint8_t dtype){
	if ((lr == NULL && lri == NULL) || (beta == NULL && betai == NULL) || ctx == NULL || net == NULL) return NULL;
	else if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	uint32_t dl = ctx->dtype_length[dtype];
	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	if (opti == NULL) return NULL;

	void *data = calloc(4, dl);
	if (data == NULL) {
		free(opti);
		return NULL;
	}

	if (lr != NULL)	memcpy(data, lr, dl);
	if (lri !=  NULL) memcpy(data + dl, lri, dl);

	if (beta != NULL) memcpy(data + (dl << 1), beta, dl);
	if (betai != NULL) memcpy(data+dl*3, betai, dl);

	uint32_t nneur = net->nneur;
	wmatrix ***velocity = (wmatrix***) calloc(nneur, sizeof(wmatrix**));
	if (velocity == NULL) goto optim_gdm_fail;

	wneuron neuron_tmp;
	wmatrix weight_tmp;
	uint64_t layers;
	uint8_t bias;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		velocity[i] = (wmatrix**) calloc(layers, sizeof(wmatrix*));
		if (velocity[i] == NULL) goto optim_gdm_fail;

		bias = 0;
		if (neuron_tmp->bias != NULL) bias = 1;

		for (uint64_t j = 0; j < layers; j++){
			weight_tmp = neuron_tmp->weight[j];
			velocity[i][j] = calloc(bias+1, sizeof(wmatrix));
			if (velocity[i][j] == NULL) goto optim_gdm_fail;

			velocity[i][j][0] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
			if (velocity[i][j][0] == NULL) goto optim_gdm_fail;

			if (bias){
				weight_tmp = neuron_tmp->bias[j];
				velocity[i][j][1] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
				if (velocity[i][j][1] == NULL) goto optim_gdm_fail;
			}
		}
	}

	opti->net = net;
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->params = data;
	opti->others = velocity;
	opti->step = &step_optim_gdm;
	opti->zero = &zero_optim_gdm;
	opti->free_func = &free_optim_gdm;

	goto optim_gdm_success;

	optim_gdm_fail:
	if (velocity != NULL){
		for (uint32_t i = 0; i < nneur; i++){
			neuron_tmp = net->neurons[i];
			layers = neuron_tmp->layer;
			if (velocity[i] == NULL) break;
			for (uint64_t j = 0; j < layers; j++){
				if (velocity[i][j] == NULL) break;
				wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
				free(velocity[i][j]);
			}
			free(velocity[i]);
		}
		free(velocity);
	}

	free(data);
	optim_gdm_success:
	

	return opti;
}

// Nesterov Accelerated Gradient

int step_optim_nag(void *data, void *grad, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weight, wmatrix bias){
	void *lr = data;
	void *lri = data + dl;
	void *beta = data + (dl << 1);
	void *betai = data + dl + (dl << 1);
	uint8_t mode = ((uint8_t*)(data + (dl << 2)))[0];
	uint32_t st, to;
	st = ((uint32_t*)(data + (dl << 2) + 1))[0];
	to = ((uint32_t*)(data + (dl << 2) + 5))[0];

	wmatrix *gradient = grad;
	wmatrix w_fake = gradient[1];

	wekuaContext ctx = w_fake->ctx;
	cl_command_queue cmd = ctx->command_queue;
	
	uint32_t evn = 0;
	uint64_t size;
	cl_event e[6];
	int ret;

	exchange_ptr(w_fake, weight);
	if (mode){
		ret = wekuaBlasScalar(gradient[0], beta, betai, 0, NULL, e);
		if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
		evn++;

		ret = wekuaBlasAxpy(error, gradient[0], lr, lri, 1, e, &e[1]);
		if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
		evn++;

		ret = wekuaMatrixAdd(weight, gradient[0], 1, &e[1], &e[2]);
		if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
		evn++;

		if (error_b != NULL){
			exchange_ptr(gradient[3], bias);
			ret = wekuaBlasScalar(gradient[2], beta, betai, 0, NULL, &e[3]);
			if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
			evn++;

			ret = wekuaBlasAxpy(error_b, gradient[2], lr, lri, 1, &e[3], &e[4]);
			if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
			evn++;

			ret = wekuaMatrixAdd(bias, gradient[2], 1, &e[4], &e[5]);
			if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
			evn++;
		}
	}else{
		size = weight->size;
		ret = clEnqueueCopyBuffer(cmd, w_fake->real, weight->real, 0, 0, size, 0, NULL, e);
		if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
		evn++;

		if (weight->com){
			if (createComplexMatrix(w_fake)) goto wk_step_optim_nag_fail;
			
			ret = clEnqueueCopyBuffer(cmd, w_fake->imag, weight->imag, 0, 0, size, 1, e, &e[1]);
			if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
			evn++;
		}
		ret = wekuaBlasAxpy(gradient[0], weight, lr, lri, 1, &e[evn-1], &e[evn]);

		if (error_b != NULL){
			w_fake = gradient[3];
			exchange_ptr(w_fake, bias);

			size = bias->size;
			ret = clEnqueueCopyBuffer(cmd, w_fake->real, bias->real, 0, 0, size, 0, NULL, &e[evn]);
			if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
			evn++;

			if (weight->com){
				if (createComplexMatrix(w_fake)) goto wk_step_optim_nag_fail;
				
				ret = clEnqueueCopyBuffer(cmd, w_fake->imag, bias->imag, 0, 0, size, 1, &e[evn-1], &e[evn]);
				if (ret != CL_SUCCESS) goto wk_step_optim_nag_fail;
				evn++;
			}
			ret = wekuaBlasAxpy(gradient[2], bias, lr, lri, 1, &e[evn-1], &e[evn]);
		}
	}

	st++;
	if (st == to){
		((uint8_t*)(data + (dl << 2)))[0] = mode^1;
		st = 0;
	}
	((uint32_t*)(data + (dl << 2) + 1))[0] = st;

	wk_step_optim_nag_fail:
	clWaitForEvents(evn, e);
	for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);

	return ret;
}

int zero_optim_nag(void *opti){
	woptim optim = opti;
	wnetwork net = optim->net;
	wneuron neuron_tmp;
	wmatrix ***velocity = optim->others;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			zero_optim(velocity[i][j][0]);
			zero_optim(velocity[i][j][1]);
			zero_optim(velocity[i][j][2]);
			zero_optim(velocity[i][j][3]);
		}
	}
	return CL_SUCCESS;
}


void free_optim_nag(void *opti, uint32_t nw, cl_event *be){
	if (opti == NULL) return;
	clWaitForEvents(nw, be);
	woptim optim = opti;

	wnetwork net = optim->net;
	wmatrix ***velocity = optim->others;
	wneuron neuron_tmp;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	free(optim->params);

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][2], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][3], 0, NULL);
			free(velocity[i][j]);
		}
		free(velocity[i]);
	}
	free(velocity);
	free(opti);
}


woptim wekuaOptimNAG(wekuaContext ctx, wnetwork net, void *lr, void *lri, void *beta, void *betai, uint8_t dtype){
	if ((lr == NULL && lri == NULL) || (beta == NULL && betai == NULL) || ctx == NULL || net == NULL) return NULL;
	else if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	uint32_t dl = ctx->dtype_length[dtype];
	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	if (opti == NULL) {
		return NULL;
	}


	void *data = malloc((dl << 2) + 9);
	if (data == NULL) {
		free(opti);
		return NULL;
	}

	memset(data, 0, (dl<<2)+9);

	if (lr != NULL) memcpy(data, lr, dl);
	if (lri != NULL) memcpy(data + dl, lri, dl);

	if (beta != NULL) memcpy(data + (dl << 1), beta, dl);
	if (betai != NULL) memcpy(data + dl*3, betai, dl);

	uint32_t nneur = net->nneur;
	memcpy(data + (dl << 2) + 5, &nneur, 4);

	wmatrix ***velocity = (wmatrix***) calloc(nneur, sizeof(wmatrix**));
	if (velocity == NULL) goto optim_nag_fail;

	wneuron neuron_tmp;
	wmatrix weight_tmp;
	uint64_t layers;
	uint8_t bias;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		velocity[i] = (wmatrix**) calloc(layers, sizeof(wmatrix*));
		if (velocity[i] == NULL) goto optim_nag_fail;

		bias = 0;
		if (neuron_tmp->bias != NULL) bias = 1;

		for (uint64_t j = 0; j < layers; j++){
			weight_tmp = neuron_tmp->weight[j];
			velocity[i][j] = calloc((size_t)(2*(bias + 1)), sizeof(wmatrix));
			if (velocity[i][j] == NULL) goto optim_nag_fail;

			velocity[i][j][0] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
			if (velocity[i][j][0] == NULL) goto optim_nag_fail;

			velocity[i][j][1] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
			if (velocity[i][j][1] == NULL) goto optim_nag_fail;
			if (bias){
				weight_tmp = neuron_tmp->bias[j];
				velocity[i][j][2] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
				if (velocity[i][j][2] == NULL) goto optim_nag_fail;

				velocity[i][j][3] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
				if (velocity[i][j][3] == NULL) goto optim_nag_fail;
			}
		}
	}

	opti->net = net;
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->params = data;
	opti->others = velocity;
	opti->step = &step_optim_nag;
	opti->zero = &zero_optim_nag;
	opti->free_func = &free_optim_nag;

	goto optim_gdm_success;

	optim_nag_fail:
	if (velocity != NULL){
		for (uint32_t i = 0; i < nneur; i++){
			neuron_tmp = net->neurons[i];
			layers = neuron_tmp->layer;
			if (velocity[i] == NULL) break;
			for (uint64_t j = 0; j < layers; j++){
				if (velocity[i][j] == NULL) break;
				wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][2], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][3], 0, NULL);
				free(velocity[i][j]);
			}
			free(velocity[i]);
		}
		free(velocity);
	}

	optim_gdm_success:
	free(data);
	return opti;
}


// Adaptive gradient optimization

int step_optim_adagrad(void *data, void *grad, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weight, wmatrix bias){
	void *lr = data;
	void *lri = data + dl;

	int ret;
	wekuaContext ctx = error->ctx;
	uint8_t dtype = error->dtype, com = weight->com;
	uint32_t evn = 0;
	cl_event e[2];

	wmatrix *gradient = grad;

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_ADAGRAD, dtype, com);
	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &error->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &error->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &weight->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &gradient[0]->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &gradient[0]->imag);
	clSetKernelArg(kernel, 6, dl, lr);
	clSetKernelArg(kernel, 7, dl, lri);
	clSetKernelArg(kernel, 8, 8, &weight->vl_shape[1]);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, weight->vl_shape, weight->work_items, 0, NULL, e);
	if (ret != CL_SUCCESS) goto wk_step_optim_adagrad_fail;
	evn++;

	if (error_b != NULL){
		if (com){
			if (createComplexMatrix(bias)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &error_b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &error_b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias->real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias->imag);
		clSetKernelArg(kernel, 4, sizeof(cl_mem), &gradient[1]->real);
		clSetKernelArg(kernel, 5, sizeof(cl_mem), &gradient[1]->imag);
		clSetKernelArg(kernel, 6, dl, lr);
		clSetKernelArg(kernel, 7, dl, lri);
		clSetKernelArg(kernel, 8, 8, &bias->vl_shape[1]);

		ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, bias->vl_shape, bias->work_items, 0, NULL, &e[1]);
		if (ret != CL_SUCCESS) goto wk_step_optim_adagrad_fail;
		evn++;
	}

	wk_step_optim_adagrad_fail:
	if (evn > 0){
		clWaitForEvents(evn, e);
		for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);
	}
	return ret;
}

int zero_optim_adagrad(void *opti){
	woptim optim = opti;
	wnetwork net = optim->net;
	wneuron neuron_tmp;
	wmatrix ***velocity = optim->others;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			zero_optim(velocity[i][j][0]);
			zero_optim(velocity[i][j][1]);
		}
	}
	return CL_SUCCESS;
}

void free_optim_adagrad(void *opti, uint32_t nw, cl_event *be){
	if (opti == NULL) return;
	clWaitForEvents(nw, be);
	woptim optim = opti;

	wnetwork net = optim->net;
	wmatrix ***velocity = optim->others;
	wneuron neuron_tmp;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	free(optim->params);

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
			free(velocity[i][j]);
		}
		free(velocity[i]);
	}
	free(velocity);
	free(opti);
}

woptim wekuaOptimAdaGrad(wekuaContext ctx, wnetwork net, void *lr, void *lri, uint8_t dtype){
	if ((lr == NULL && lri == NULL) || ctx == NULL || net == NULL) return NULL;
	else if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	if (opti == NULL) return NULL;

	uint32_t dl = ctx->dtype_length[dtype];
	void *data = calloc(2, dl);
	if (data == NULL) {
		free(opti);
		return NULL;
	}

	if (lr != NULL) memcpy(data, lr, dl);
	if (lri != NULL) memcpy(data+dl, lri, dl);

	uint32_t nneur = net->nneur;
	wmatrix ***velocity = (wmatrix***) calloc(nneur, sizeof(wmatrix**));
	if (velocity == NULL) goto optim_adagrad_fail;

	wneuron neuron_tmp;
	wmatrix weight_tmp;
	uint64_t layers;
	uint8_t bias;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		velocity[i] = (wmatrix**) calloc(layers, sizeof(wmatrix*));
		if (velocity[i] == NULL) goto optim_adagrad_fail;

		bias = 0;
		if (neuron_tmp->bias != NULL) bias = 1;

		for (uint64_t j = 0; j < layers; j++){
			weight_tmp = neuron_tmp->weight[j];
			velocity[i][j] = calloc((size_t)(2*(bias+1)), sizeof(wmatrix));
			if (velocity[i][j] == NULL) goto optim_adagrad_fail;

			velocity[i][j][0] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
			if (velocity[i][j][0] == NULL) goto optim_adagrad_fail;
			if (bias){
				weight_tmp = neuron_tmp->bias[j];

				velocity[i][j][1] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
				if (velocity[i][j][1] == NULL) goto optim_adagrad_fail;
			}
		}
	}

	opti->net = net;
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->params = data;
	opti->others = velocity;
	opti->step = &step_optim_adagrad;
	opti->zero = &zero_optim_adagrad;
	opti->free_func = &free_optim_adagrad;

	goto optim_gdm_success;

	optim_adagrad_fail:
	if (velocity != NULL){
		for (uint32_t i = 0; i < nneur; i++){
			neuron_tmp = net->neurons[i];
			layers = neuron_tmp->layer;
			if (velocity[i] == NULL) break;

			for (uint64_t j = 0; j < layers; j++){
				if (velocity[i][j] == NULL) break;
				wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
				free(velocity[i][j]);
			}
			free(velocity[i]);
		}
		free(velocity);
	}

	free(data);
	optim_gdm_success:
	return opti;
}

// Root Mean Square Propagation

int step_optim_rmsprop(void *data, void *grad, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weight, wmatrix bias){
	void *lr = data;
	void *lri = data + dl;
	void *betar = data + (dl << 1);
	void *betai = data + dl*3;

	int ret;
	wekuaContext ctx = error->ctx;
	uint8_t dtype = error->dtype, com = weight->com;
	uint32_t evn = 0;
	cl_event e[2];

	wmatrix *gradient = grad;

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_RMSPROP, dtype, com);
	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &error->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &error->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &weight->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &gradient[0]->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &gradient[0]->imag);
	clSetKernelArg(kernel, 6, dl, lr);
	clSetKernelArg(kernel, 7, dl, lri);
	clSetKernelArg(kernel, 8, dl, betar);
	clSetKernelArg(kernel, 9, dl, betai);
	clSetKernelArg(kernel, 10, 8, &weight->vl_shape[1]);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, weight->vl_shape, weight->work_items, 0, NULL, e);
	if (ret != CL_SUCCESS) goto wk_step_optim_adagrad_fail;
	evn++;

	if (error_b != NULL){
		if (com){
			if (createComplexMatrix(bias)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &error_b->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &error_b->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias->real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias->imag);
		clSetKernelArg(kernel, 4, sizeof(cl_mem), &gradient[1]->real);
		clSetKernelArg(kernel, 5, sizeof(cl_mem), &gradient[1]->imag);
		clSetKernelArg(kernel, 6, dl, lr);
		clSetKernelArg(kernel, 7, dl, lri);
		clSetKernelArg(kernel, 8, dl, betar);
		clSetKernelArg(kernel, 9, dl, betai);
		clSetKernelArg(kernel, 10, 8, &bias->vl_shape[1]);

		ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, bias->vl_shape, bias->work_items, 0, NULL, &e[1]);
		if (ret != CL_SUCCESS) goto wk_step_optim_adagrad_fail;
		evn++;
	}

	wk_step_optim_adagrad_fail:
	if (evn > 0){
		clWaitForEvents(evn, e);
		for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);
	}
	return ret;
}

int zero_optim_rmsprop(void *opti){
	woptim optim = opti;
	wnetwork net = optim->net;
	wneuron neuron_tmp;
	wmatrix ***velocity = optim->others;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			zero_optim(velocity[i][j][0]);
			zero_optim(velocity[i][j][1]);
		}
	}
	return CL_SUCCESS;
}

void free_optim_rmsprop(void *opti, uint32_t nw, cl_event *be){
	if (opti == NULL) return;
	clWaitForEvents(nw, be);
	woptim optim = opti;

	wnetwork net = optim->net;
	wmatrix ***velocity = optim->others;
	wneuron neuron_tmp;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	free(optim->params);

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
			free(velocity[i][j]);
		}
		free(velocity[i]);
	}
	free(velocity);
	free(opti);
}

woptim wekuaOptimRMSProp(wekuaContext ctx, wnetwork net, void *lr, void *lri, void *beta, void *betai, uint8_t dtype){
	if ((lr == NULL && lri == NULL) || (beta == NULL && betai == NULL) || ctx == NULL || net == NULL) return NULL;
	else if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	if (opti == NULL) return NULL;

	uint32_t dl = ctx->dtype_length[dtype];
	void *data = calloc(4, dl);
	if (data == NULL) {
		free(opti);
		return NULL;
	}

	if (lr != NULL) memcpy(data, lr, dl);
	if (beta != NULL) memcpy(data+(dl<<1), beta, dl);
	
	if (lri != NULL) memcpy(data+dl, lri, dl);
	if (betai != NULL) memcpy(data+dl*3, betai, dl);

	uint32_t nneur = net->nneur;
	wmatrix ***velocity = (wmatrix***) calloc(nneur, sizeof(wmatrix**));
	if (velocity == NULL) goto optim_rmsprop_fail;
	wneuron neuron_tmp;
	wmatrix weight_tmp;
	uint64_t layers;
	uint8_t bias;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		velocity[i] = (wmatrix**) calloc(layers, sizeof(wmatrix*));
		if (velocity[i] == NULL) goto optim_rmsprop_fail;
		bias = 0;
		if (neuron_tmp->bias != NULL) bias = 1;

		for (uint64_t j = 0; j < layers; j++){
			weight_tmp = neuron_tmp->weight[j];
			velocity[i][j] = calloc((size_t)(2*(bias+1)), sizeof(wmatrix));
			if (velocity[i][j] == NULL) goto optim_rmsprop_fail;

			velocity[i][j][0] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
			if (velocity[i][j][0] == NULL) goto optim_rmsprop_fail;
			if (bias){
				weight_tmp = neuron_tmp->bias[j];

				velocity[i][j][1] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
				if (velocity[i][j][1] == NULL) goto optim_rmsprop_fail;
			}
		}
	}

	opti->net = net;
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->params = data;
	opti->others = velocity;
	opti->step = &step_optim_rmsprop;
	opti->zero = &zero_optim_rmsprop;
	opti->free_func = &free_optim_rmsprop;

	goto optim_gdm_success;

	optim_rmsprop_fail:
	if (velocity != NULL){
		for (uint32_t i = 0; i < nneur; i++){
			neuron_tmp = net->neurons[i];
			layers = neuron_tmp->layer;
			if (velocity[i] == NULL) break;
			for (uint64_t j = 0; j < layers; j++){
				if (velocity[i][j] == NULL) break;
				wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
				free(velocity[i][j]);
			}
			free(velocity[i]);
		}
		free(velocity);
	}

	optim_gdm_success:
	free(data);
	return opti;
}


// Adadelta

int step_optim_adadelta(void *data, void *grad, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weight, wmatrix bias){
	void *lr = data;
	void *lri = data + dl;

	int ret;
	wekuaContext ctx = error->ctx;
	uint8_t dtype = error->dtype, com = weight->com;
	uint32_t evn = 0;
	cl_event e[2];

	wmatrix *gradient = grad;

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_ADADELTA, dtype, com);
	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &gradient[0]->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient[0]->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient[1]->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &gradient[1]->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &error->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &error->imag);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &weight->real);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &weight->imag);
	clSetKernelArg(kernel, 8, dl, lr);
	clSetKernelArg(kernel, 9, dl, lri);
	clSetKernelArg(kernel, 10, 8, &weight->vl_shape[1]);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, weight->vl_shape, weight->work_items, 0, NULL, e);
	if (ret != CL_SUCCESS) goto wk_step_optim_adadelta_fail;
	evn++;

	if (error_b != NULL){
		if (com){
			if (createComplexMatrix(bias)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &gradient[2]->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient[2]->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient[3]->real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &gradient[3]->imag);
		clSetKernelArg(kernel, 4, sizeof(cl_mem), &error_b->real);
		clSetKernelArg(kernel, 5, sizeof(cl_mem), &error_b->imag);
		clSetKernelArg(kernel, 6, sizeof(cl_mem), &bias->real);
		clSetKernelArg(kernel, 7, sizeof(cl_mem), &bias->imag);
		clSetKernelArg(kernel, 8, dl, lr);
		clSetKernelArg(kernel, 9, dl, lri);
		clSetKernelArg(kernel, 10, 8, &bias->vl_shape[1]);

		ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, bias->vl_shape, bias->work_items, 0, NULL, &e[1]);
		if (ret != CL_SUCCESS) goto wk_step_optim_adadelta_fail;
		evn++;
	}

	wk_step_optim_adadelta_fail:
	if (evn > 0){
		clWaitForEvents(evn, e);
		for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);
	}
	return ret;
}

int zero_optim_adadelta(void *opti){
	woptim optim = opti;
	wnetwork net = optim->net;
	wneuron neuron_tmp;
	wmatrix ***velocity = optim->others;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			zero_optim(velocity[i][j][0]);
			zero_optim(velocity[i][j][1]);
			zero_optim(velocity[i][j][2]);
			zero_optim(velocity[i][j][3]);
		}
	}
	return CL_SUCCESS;
}

void free_optim_adadelta(void *opti, uint32_t nw, cl_event *be){
	if (opti == NULL) return;
	clWaitForEvents(nw, be);
	woptim optim = opti;

	wnetwork net = optim->net;
	wmatrix ***velocity = optim->others;
	wneuron neuron_tmp;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	free(optim->params);

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][2], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][3], 0, NULL);
			free(velocity[i][j]);
		}
		free(velocity[i]);
	}
	free(velocity);
	free(opti);
}

woptim wekuaOptimAdadelta(wekuaContext ctx, wnetwork net, void *lr, void *lri, uint8_t dtype){
	if ((lr == NULL && lri == NULL) || ctx == NULL || net == NULL) return NULL;
	else if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	if (opti == NULL) return NULL;

	uint32_t dl = ctx->dtype_length[dtype];
	void *data = calloc(2, dl);
	if (data == NULL) {
		free(opti);
		return NULL;
	}

	void *one = get_one(dtype, dl);

	if (lr != NULL) memcpy(data, lr, dl);
	if (lri != NULL) memcpy(data+dl, lri, dl);

	uint32_t nneur = net->nneur;
	wmatrix ***velocity = (wmatrix***) calloc(nneur, sizeof(wmatrix**));
	if (velocity == NULL) goto optim_adadelta_fail;

	wneuron neuron_tmp;
	wmatrix weight_tmp;
	uint64_t layers;
	uint8_t bias;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		assert(layers > 0);

		velocity[i] = (wmatrix**) calloc(layers, sizeof(wmatrix*));
		if (velocity[i] == NULL) goto optim_adadelta_fail;

		bias = 0;
		if (neuron_tmp->bias != NULL) bias = 1;

		for (uint64_t j = 0; j < layers; j++){
			weight_tmp = neuron_tmp->weight[j];
			velocity[i][j] = calloc((size_t)(2*(bias+1)), sizeof(wmatrix));
			if (velocity[i][j] == NULL) goto optim_adadelta_fail;

			velocity[i][j][0] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
			if (velocity[i][j][0] == NULL) goto optim_adadelta_fail;

			velocity[i][j][1] = wekuaFillMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], one, NULL, dtype);
			if (velocity[i][j][1] == NULL) goto optim_adadelta_fail;
			if (bias){
				weight_tmp = neuron_tmp->bias[j];

				velocity[i][j][2] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
				if (velocity[i][j][2] == NULL) goto optim_adadelta_fail;

				velocity[i][j][3] = wekuaFillMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], one, NULL, dtype);
				if (velocity[i][j][3] == NULL) goto optim_adadelta_fail;
			}
		}
	}

	opti->net = net;
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->params = data;
	opti->others = velocity;
	opti->step = &step_optim_adadelta;
	opti->zero = &zero_optim_adadelta;
	opti->free_func = &free_optim_adadelta;

	goto optim_gdm_success;

	optim_adadelta_fail:
	if (velocity != NULL){
		for (uint32_t i = 0; i < nneur; i++){
			neuron_tmp = net->neurons[i];
			layers = neuron_tmp->layer;
			assert(layers > 0);

			if (velocity[i] == NULL) break;
			for (uint64_t j = 0; j < layers; j++){
				if (velocity[i][j] == NULL) break;
				wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][3], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][2], 0, NULL);
				free(velocity[i][j]);
			}
			free(velocity[i]);
		}
		free(velocity);
	}

	optim_gdm_success:
	free(data);
	free(one);
	return opti;
}

int step_optim_adam(void *data, void *grad, uint32_t dl, wmatrix error, wmatrix error_b, wmatrix weight, wmatrix bias){
	void *lr = data;
	void *lri = data + dl;
	void *beta1r = lri + dl;
	void *beta1i = beta1r + dl;
	void *beta2r = beta1i + dl;
	void *beta2i = beta2r + dl;

	int ret;
	wekuaContext ctx = error->ctx;
	uint8_t dtype = error->dtype, com = weight->com;
	uint32_t evn = 0;
	cl_event e[2];

	wmatrix *gradient = grad;

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_ADADELTA, dtype, com);
	if (kernel == NULL) return CL_BUILD_PROGRAM_FAILURE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &gradient[0]->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient[0]->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient[1]->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &gradient[1]->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &error->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &error->imag);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &weight->real);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &weight->imag);
	clSetKernelArg(kernel, 8, dl, lr);
	clSetKernelArg(kernel, 9, dl, lri);
	clSetKernelArg(kernel, 10, dl, beta1r);
	clSetKernelArg(kernel, 11, dl, beta1i);
	clSetKernelArg(kernel, 12, dl, beta2r);
	clSetKernelArg(kernel, 13, dl, beta2i);
	clSetKernelArg(kernel, 14, 8, &weight->vl_shape[1]);

	ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, weight->vl_shape, weight->work_items, 0, NULL, e);
	if (ret != CL_SUCCESS) goto wk_step_optim_adadelta_fail;
	evn++;

	if (error_b != NULL){
		if (com){
			if (createComplexMatrix(bias)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
		}

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &gradient[2]->real);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient[2]->imag);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient[3]->real);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &gradient[3]->imag);
		clSetKernelArg(kernel, 4, sizeof(cl_mem), &error_b->real);
		clSetKernelArg(kernel, 5, sizeof(cl_mem), &error_b->imag);
		clSetKernelArg(kernel, 6, sizeof(cl_mem), &bias->real);
		clSetKernelArg(kernel, 7, sizeof(cl_mem), &bias->imag);
		clSetKernelArg(kernel, 8, dl, lr);
		clSetKernelArg(kernel, 9, dl, lri);
		clSetKernelArg(kernel, 10, dl, beta1r);
		clSetKernelArg(kernel, 11, dl, beta1i);
		clSetKernelArg(kernel, 12, dl, beta2r);
		clSetKernelArg(kernel, 13, dl, beta2i);
		clSetKernelArg(kernel, 14, 8, &bias->vl_shape[1]);

		ret = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL, bias->vl_shape, bias->work_items, 0, NULL, &e[1]);
		if (ret != CL_SUCCESS) goto wk_step_optim_adadelta_fail;
		evn++;
	}

	wk_step_optim_adadelta_fail:
	if (evn > 0){
		clWaitForEvents(evn, e);
		for (uint32_t x=0; x<evn; x++) clReleaseEvent(e[x]);
	}
	return ret;
}

int zero_optim_adam(void *opti){
	woptim optim = opti;
	wnetwork net = optim->net;
	wneuron neuron_tmp;
	wmatrix ***velocity = optim->others;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			zero_optim(velocity[i][j][0]);
			zero_optim(velocity[i][j][1]);
			zero_optim(velocity[i][j][2]);
			zero_optim(velocity[i][j][3]);
		}
	}
	return CL_SUCCESS;
}

void free_optim_adam(void *opti, uint32_t nw, cl_event *be){
	if (opti == NULL) return;
	clWaitForEvents(nw, be);
	woptim optim = opti;

	wnetwork net = optim->net;
	wmatrix ***velocity = optim->others;
	wneuron neuron_tmp;
	uint32_t nneur = net->nneur;
	uint64_t layers;

	free(optim->params);

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		if (velocity[i] == NULL) break;
		for (uint64_t j = 0; j < layers; j++){
			if (velocity[i][j] == NULL) break;
			wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][2], 0, NULL);
			wekuaFreeMatrix(velocity[i][j][3], 0, NULL);
			free(velocity[i][j]);
		}
		free(velocity[i]);
	}
	free(velocity);
	free(opti);
}

woptim wekuaOptimAdam(wekuaContext ctx,  wnetwork net, void *lr, void *lri, void *beta1, void *beta1i, void *beta2, void *beta2i, uint8_t dtype){
	if ((lr == NULL && lri == NULL) || (beta1 == NULL && beta1i == NULL) || (beta2 == NULL && beta2i == NULL) || ctx == NULL || net == NULL) return NULL;
	else if (dtype < WEKUA_DTYPE_FLOAT) return NULL;

	woptim opti = (woptim) calloc(1, sizeof(struct _w_optim));
	if (opti == NULL) return NULL;

	uint32_t dl = ctx->dtype_length[dtype];
	void *data = calloc(6, dl);
	if (data == NULL) {
		free(opti);
		return NULL;
	}

	if (lr != NULL) memcpy(data, lr, dl);
	if (lri != NULL) memcpy(data+dl, lri, dl);

	if (beta1 != NULL) memcpy(data + (dl << 1), beta1, dl);
	if (beta1i != NULL) memcpy(data+3*dl, beta1i, dl);

	if (beta2 != NULL) memcpy(data + (dl << 2), beta2, dl);
	if (beta2i != NULL) memcpy(data+5*dl, beta2i, dl);

	uint32_t nneur = net->nneur;
	wmatrix ***velocity = (wmatrix***) calloc(nneur, sizeof(wmatrix**));
	if (velocity == NULL) goto optim_adam_fail;

	wneuron neuron_tmp;
	wmatrix weight_tmp;
	uint64_t layers;
	uint8_t bias;

	for (uint32_t i = 0; i < nneur; i++){
		neuron_tmp = net->neurons[i];
		layers = neuron_tmp->layer;
		assert(layers > 0);

		velocity[i] = (wmatrix**) calloc(layers, sizeof(wmatrix*));
		if (velocity[i] == NULL) goto optim_adam_fail;

		bias = 0;
		if (neuron_tmp->bias != NULL) bias = 1;

		for (uint64_t j = 0; j < layers; j++){
			velocity[i][j] = calloc((size_t)(2*(bias+1)), sizeof(wmatrix));
			if (velocity[i][j] == NULL) goto optim_adam_fail;

			weight_tmp = neuron_tmp->weight[j];

			velocity[i][j][0] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
			if (velocity[i][j][0] == NULL) goto optim_adam_fail;

			velocity[i][j][1] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
			if (velocity[i][j][1] == NULL) goto optim_adam_fail;
			if (bias){
				weight_tmp = neuron_tmp->bias[j];

				velocity[i][j][2] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
				if (velocity[i][j][2] == NULL) goto optim_adam_fail;

				velocity[i][j][3] = wekuaAllocMatrix(ctx, weight_tmp->shape[0], weight_tmp->shape[1], dtype);
				if (velocity[i][j][3] == NULL) goto optim_adam_fail;
			}
		}
	}

	opti->net = net;
	opti->dtype = dtype;
	opti->ctx = ctx;
	opti->params = data;
	opti->others = velocity;
	opti->step = &step_optim_adam;
	opti->zero = &zero_optim_adam;
	opti->free_func = &free_optim_adam;

	goto optim_adam_success;

	optim_adam_fail:
	if (velocity != NULL){
		for (uint32_t i = 0; i < nneur; i++){
			neuron_tmp = net->neurons[i];
			layers = neuron_tmp->layer;
			assert(layers > 0);

			if (velocity[i] == NULL) break;

			for (uint64_t j = 0; j < layers; j++){
				if (velocity[i][j] == NULL) break;
				wekuaFreeMatrix(velocity[i][j][0], 0, NULL);
				wekuaFreeMatrix(velocity[i][j][1], 0, NULL);
				if (neuron_tmp->bias) {
					wekuaFreeMatrix(velocity[i][j][2], 0, NULL);
					wekuaFreeMatrix(velocity[i][j][3], 0, NULL);
				}
				free(velocity[i][j]);
			}
			free(velocity[i]);
		}
		free(velocity);
	}

	optim_adam_success:
	free(data);
	return opti;
}





static wmatrix get_dev_bias(wekuaContext ctx, wmatrix error, uint8_t dtype){
	cl_event e;
	uint8_t com = error->com;
	uint64_t row, wi;

	row = error->shape[0];

	cl_kernel kernel = compileKernel(ctx, WEKUA_KERNEL_LINEAR_BIAS_STEP, dtype, com);
	if (kernel == NULL) return NULL;

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

static int step(wneuron neuron, void *opti_data, wmatrix **grad, werror error, wcache cache, int (*func)(void *, void *, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix)){
	uint64_t layers = neuron->layer, x = 0;
	uint8_t dtype = neuron->dtype;
	uint32_t dl;
	cl_event event;
	int ret = CL_SUCCESS;
	
	wekuaContext ctx;
	wmatrix *weigth = neuron->weight;
	wmatrix *bias = neuron->bias;
	wmatrix *cache_d = cache->data;
	wmatrix *errors = error->o_err;

	wmatrix dev, dev_bias = NULL;
	ctx = weigth[0]->ctx;
	dl = ctx->dtype_length[dtype];

	void *one = get_one(dtype, dl);
	wmatrix w, e, a, *g = NULL;
	for (; x<layers; x++){
		w = weigth[x];
		a = cache_d[x];

		if (grad != NULL) g = grad[x];

		e = wekuaMatrixTrans(errors[x], 0, NULL, &event);
		if (e == NULL) break;

		dev = wekuaAllocMatrix(ctx, w->shape[0], w->shape[1], dtype);		
		ret = wekuaBlasGemm(one, NULL, 0, e, 0, a, NULL, NULL, dev, 1, &event);
		if (ret != CL_SUCCESS){
			clWaitForEvents(1, &event);
			wekuaFreeMatrix(dev, 0, NULL);
			wekuaFreeMatrix(e, 0, NULL);
			break;
		}
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

int wekuaOptimStep(woptim optim, werror *error, wcache *cache){
	if (optim == NULL || error == NULL || cache == NULL) return CL_INVALID_ARG_VALUE;

	int ret = CL_SUCCESS;
	wnetwork net = optim->net;
	wneuron *neurons = net->neurons;
	uint32_t nneur = net->nneur;
	uint32_t x = 0;
	void *params = optim->params;
	wmatrix ***others = optim->others;
	int (*func)(void *, void*, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix);

	func = optim->step;

	wmatrix **grad = NULL;

	for (; x<nneur; x++){
		if (others != NULL) grad = others[x];

		ret = step(neurons[x], params, grad, error[nneur - x - 1], cache[x], func);
		if (ret != CL_SUCCESS) break;
	}

	return ret;
}

int wekuaOptimZero(woptim optim){
	if (optim == NULL) return CL_INVALID_ARG_VALUE;
	if (optim->zero == NULL) return CL_SUCCESS;
	return optim->zero(optim->others);
}

void wekuaFreeOptim(woptim optim, uint32_t nw, cl_event *be){
	if (optim == NULL) return;

	optim->free_func(optim, nw, be);
}

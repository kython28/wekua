#include "../headers/neuron.h"

wneuron wekuaLinear(wekuaContext ctx, uint64_t input, uint64_t output, uint64_t deep, uint8_t bias, wacti acti, uint8_t dtype){
	if (input == 0 || output == 0 || deep == 0 || acti == NULL || dtype < WEKUA_DTYPE_FLOAT || dtype > WEKUA_DTYPE_DOUBLE) return NULL;

	wmatrix tmp, *weight = NULL, *bias_w = NULL;
	cl_event e;

	wneuron neur = (wneuron) calloc(1, sizeof(struct _w_neuron));
	if (neur == NULL) return NULL;

	neur->layer = deep;
	neur->dtype = dtype;
	neur->acti = acti;

	weight = (wmatrix*) calloc(deep, sizeof(wmatrix));
	if (weight == NULL) return NULL;
	neur->weight = weight;

	void *start, *end;
	if (dtype == WEKUA_DTYPE_FLOAT){
		start = malloc(sizeof(float));
		end = malloc(sizeof(float));

		((float*)start)[0] = -1.0;
		((float*)end)[0] = 1.0;
	}else{
		start = malloc(sizeof(double));
		end = malloc(sizeof(double));

		((double*)start)[0] = -1.0;
		((double*)end)[0] = 1.0;
	}

	weight[0] = wekuaMatrixRandUniform(ctx, output, input, start, NULL, end, NULL, dtype);
	if (weight[0] == NULL) goto wekua_linear_fail;

	for (uint64_t x=1; x<deep; x++){
		weight[x] = wekuaMatrixRandUniform(ctx, output, output, start, NULL, end, NULL, dtype);
		if (weight[x] == NULL) goto wekua_linear_fail;
	}

	if (bias){
		bias_w = (wmatrix*) calloc(deep, sizeof(wmatrix));
		if (bias_w == NULL) goto wekua_linear_fail;
		neur->bias = bias_w;

		for (uint64_t x=0; x<deep; x++){
			bias_w[x] = wekuaMatrixRandUniform(ctx, 1, output, start, NULL, end, NULL, dtype);
			if (bias_w[x] == NULL) goto wekua_linear_fail;
		}
	}

	goto wekua_linear_success;

	wekua_linear_fail:
	if (weight != NULL){
		for (uint64_t x=0; x<deep; x++){
			wekuaFreeMatrix(weight[x], 0, NULL);
		}
		free(weight);
	}

	if (bias_w != NULL){
		for (uint64_t x=0; x<deep; x++){
			wekuaFreeMatrix(bias_w[x], 0, NULL);
		}
		free(bias_w);
	}

	free(neur);
	neur = NULL;

	wekua_linear_success:
	free(start);
	free(end);

	return neur;
}
#include "../headers/neuron.h"
#include <math.h>

#define SET_LIMIT(dtype, input, output, start, end) \
	if (dtype == WEKUA_DTYPE_FLOAT){ \
		const float limit = sqrtf(6.0f/((float) (input+output))); \
		((float*)start)[0] = -limit; \
		((float*)end)[0] = limit; \
	}else{ \
		const double limit = sqrt(6.0/((double) (input+output))); \
		((double*)start)[0] = -limit; \
		((double*)end)[0] = limit; \
	} \

wneuron wekuaLinear(wekuaContext ctx, uint64_t input, uint64_t output, uint64_t deep, uint8_t bias, wacti acti, uint8_t dtype){
	if (input == 0 || output == 0 || deep == 0 || acti == NULL || dtype < WEKUA_DTYPE_FLOAT || dtype > WEKUA_DTYPE_DOUBLE) return NULL;

	wmatrix *weight = NULL, *bias_w = NULL;

	wneuron neur = (wneuron) calloc(1, sizeof(struct _w_neuron));
	if (neur == NULL) return NULL;

	neur->layer = deep;
	neur->dtype = dtype;
	neur->acti = acti;

	weight = (wmatrix*) calloc(deep, sizeof(wmatrix));
	if (weight == NULL) {
		free(neur);
		return NULL;
	}
	neur->weight = weight;

	void *start = NULL, *end = NULL;
	if (dtype == WEKUA_DTYPE_FLOAT){
		start = malloc(sizeof(float));
		if (start == NULL) goto wekua_linear_fail;

		end = malloc(sizeof(float));
		if (end == NULL) goto wekua_linear_fail;
	}else{
		start = malloc(sizeof(double));
		if (start == NULL) goto wekua_linear_fail;

		end = malloc(sizeof(double));
		if (end == NULL) goto wekua_linear_fail;
	}
	SET_LIMIT(dtype, input, output, start, end);

	weight[0] = wekuaMatrixRandUniform(ctx, output, input, start, NULL, end, NULL, dtype);
	if (weight[0] == NULL) goto wekua_linear_fail;

	for (uint64_t x=1; x<deep; x++){
		SET_LIMIT(dtype, output, output, start, end);
		weight[x] = wekuaMatrixRandUniform(ctx, output, output, start, NULL, end, NULL, dtype);
		if (weight[x] == NULL) goto wekua_linear_fail;
	}

	if (bias){
		bias_w = (wmatrix*) calloc(deep, sizeof(wmatrix));
		if (bias_w == NULL) goto wekua_linear_fail;
		neur->bias = bias_w;

		for (uint64_t x=0; x<deep; x++){
			bias_w[x] = wekuaAllocMatrix(ctx, 1, output, dtype);
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
	if (start != NULL) free(start);
	if (end != NULL) free(end);

	return neur;
}

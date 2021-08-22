#include "wekua.h"


void *run_conv1d(void *m, void *input_ptr, wcache *cache, uint32_t nw, cl_event *be);
int run_conv1d_bias(cl_kernel kernel, cl_command_queue queue, cl_device_local_mem_type local_mem_type, wmatrix output, wmatrix bias, uint32_t dl, cl_event *e);
int backward_conv1d(void *n, werror error, wcache cache, werror *err);
int step_conv1d(void *neur, void *opti_data, void *other, werror error, wcache cache, int (*func)(void *, void *, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix));
wmatrix get_dev_conv1d_bias(wekuaContext ctx, wmatrix error, uint8_t dtype);

void free_error_conv1d(werror err);
void free_cache_conv1d(wcache cache);

wneuron wekuaConv1d(wekuaContext ctx, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint8_t bias, wacti acti, uint8_t dtype){
	if (ctx == NULL || in_channels == 0 || out_channels == 0 || acti == NULL || kernel_size == 0 || stride == 0 || dtype < WEKUA_DTYPE_FLOAT || dtype > WEKUA_DTYPE_DOUBLE) return NULL;

	wmatrix *w, *b;

	wneuron neuron = calloc(1, sizeof(struct _w_neuron));
	if (neuron == NULL) return NULL;

	neuron->layer = out_channels;
	neuron->dtype = dtype;
	neuron->acti = acti;
	neuron->step = &step_conv1d;
	neuron->free_cache = &free_cache_conv1d;
	neuron->free_error = &free_error_conv1d;
	neuron->run = &run_conv1d;
	neuron->backward = &backward_conv1d;

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

	w = malloc(sizeof(wmatrix));
	b = malloc(sizeof(wmatrix));

	w[0] = wekuaMatrixRandUniform(ctx, out_channels, kernel_size*in_channels, start, NULL, end, NULL, dtype);
	if (w[0] == NULL) goto wekua_conv1d_fail;
	neuron->weight = w;

	b[0] = wekuaMatrixRandUniform(ctx, 1, out_channels, start, NULL, end, NULL, dtype);
	if (b[0] == NULL) goto wekua_conv1d_fail;
	neuron->bias = b;

	wekua_conv1d_fail:
	if (w != NULL) wekuaFreeMatrix(w[0], 0, NULL);
	if (b != NULL) wekuaFreeMatrix(b[0], 0, NULL);

	free(w);
	free(b);

	free(neuron);
	neuron = NULL;
	wekua_conv1d_success:
	free(start);
	free(end);
	return neuron;
}
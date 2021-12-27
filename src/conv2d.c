#include "../headers/neuron.h"

struct _w_conv2d_info {
	uint64_t row_kernel;
	uint64_t channels, stride;
};

wneuron wekuaConv2d(wekuaContext ctx, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size_w, uint64_t kernel_size_h, uint64_t stride, uint8_t bias, uint8_t dtype){
	if (ctx == NULL || in_channels == 0 || out_channels == 0 || kernel_size_w == 0 || kernel_size_h == 0 || stride == 0 || dtype < WEKUA_DTYPE_FLOAT) return NULL;
	wneuron neur = (wneuron) calloc(1, sizeof(struct _w_neuron));
	if (neur == NULL) return NULL;

	struct _w_conv2d_info *extra_data = calloc(1, sizeof(struct _w_conv2d_info));

	wmatrix *w = calloc(1, sizeof(wmatrix));
	wmatrix *b = NULL;
	if (bias){
		b = calloc(1, sizeof(wmatrix));
	}

	uint32_t wl = ctx->vector_width[dtype];

	kernel_size_w *= in_channels;
	kernel_size_w += wl - kernel_size_w%wl;
	kernel_size_w /= wl;
	kernel_size_w += 3 - kernel_size_w%3;
	kernel_size_w *= wl;

	w[0] = wekuaAllocMatrix(ctx, kernel_size_h, kernel_size_w, dtype);
	if (bias){
		b[0] = wekuaAllocMatrix(ctx, 1, out_channels, dtype);
	}

	neur->layer = 1;
	neur->dtype = dtype;
	neur->extra_data = extra_data;
	neur->weight = w;
	neur->bias = b;

	return neur;
}
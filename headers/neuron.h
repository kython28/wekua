#ifndef NEURON_H
#define NEURON_H

#include "matrix.h"
#include "cache.h"
#include "error.h"
#include "acti.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _w_neuron {
	wmatrix *weight; // Neuron weight
	wmatrix *bias; // Neuron bias
	uint64_t layer; // Layer num
	uint8_t dtype; // Weight data type
	wacti acti; // Activation function for the neuron

	void *extra_data;

	// To run the neuron
	void* (*run)(void *, void *, wcache *, uint32_t, cl_event *);
	int (*backward)(void *, werror error, wcache cache, werror *err);
	int (*step)(void *, void *, void *, werror error, wcache cache, int (*)(void *, void *, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix));

	void (*free_error)(werror);
	void (*free_cache)(wcache);
} *wneuron;

wneuron wekuaLinear(wekuaContext ctx, uint64_t input, uint64_t output, uint64_t deep, uint8_t bias, wacti acti, uint8_t dtype);

// wneuron wekuaConv1d(wekuaContext ctx, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint8_t bias, wacti acti, uint8_t dtype);
// wneuron wekuaConv2d(wekuaContext ctx, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size_w, uint64_t kernel_size_h, uint64_t stride_w, uint64_t stride_h, uint8_t bias, uint8_t dtype);

void *runWekuaNeuron(wneuron neuron, void *input, wcache *cache, uint32_t nw, cl_event *be);

int wekuaNeuronBackward(wneuron neuron, werror error, wcache cache, werror *err);

uint8_t saveWekuaNeuron(const char *name, wneuron neuron);
uint8_t loadWekuaNeuron(const char *name, wneuron neuron);

void wekuaFreeNeuron(wneuron neur, uint32_t nw, cl_event *be);

#ifdef __cplusplus
}
#endif
#endif
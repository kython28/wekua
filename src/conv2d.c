#include "../headers/neuron.h"

wneuron wekuaConv2d(wekuaContext ctx, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size_w, uint64_t kernel_size_h, uint64_t stride, uint8_t bias, wacti acti, uint8_t dtype){
	wneuron neuron = wekuaLinear(
		ctx,
		kernel_size_w*kernel_size_h*in_channels,
		out_channels,
		1,
		bias,
		acti,
		dtype
	);


	return neuron;
}
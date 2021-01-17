#include "wekua.h"

wmatrix runWekuaNeuron(wneuron neuron, wmatrix input, wcache *cache, uint32_t nw, cl_event *be){
	return neuron->run(neuron, input, cache, nw, be);
}

wnetwork wekuaNeuronNetwork(uint32_t neur_num, uint8_t dtype){
	wnetwork net = (wnetwork) calloc(1, sizeof(struct _w_net));

	net->nneur = neur_num;
	net->dtype = dtype;
	net->neurons = (wneuron*) calloc(neur_num, sizeof(wneuron));

	return net;
}

wmatrix runWekuaNetwork(wnetwork net, wmatrix input, wcache **cache){
	if (net == NULL || input == NULL) return NULL;

	wmatrix output = NULL, tmp[2] = {NULL, NULL};
	uint8_t d = 1;
	uint32_t nneur = net->nneur;
	wneuron *neurons = net->neurons;

	if (cache != NULL){
		tmp[0] = input;

		cache[0] = (wcache*) calloc(nneur, sizeof(wcache));

		for (uint32_t x=0; x<nneur; x++){
			tmp[d] = neurons[x]->run(neurons[x], tmp[d^1], &cache[0][x], 0, NULL);

			if (tmp[d] == NULL){
				output = NULL;
				break;
			}
			output = tmp[d];

			d ^= 1;
			if (tmp[d] != input) wekuaFreeMatrix(tmp[d], 0, NULL);
		}
	}else{
		tmp[0] = input;
		for (uint32_t x=0; x<nneur; x++){
			tmp[d] = neurons[x]->run(neurons[x], tmp[d^1], NULL, 0, NULL);

			if (tmp[d] == NULL){
				output = NULL;
				break;
			}
			output = tmp[d];

			d ^= 1;
			if (tmp[d] != input) wekuaFreeMatrix(tmp[d], 0, NULL);
		}
	}

	return output;
}

int wekuaNetworkBackward(wnetwork net, werror *error, wcache *cache, werror *err){
	if (net == NULL || error == NULL || cache == NULL || err == NULL) return CL_INVALID_ARG_VALUE;

	int ret;
	uint32_t nneur = net->nneur;
	uint32_t x = 0;
	wneuron *neurons = net->neurons;
	werror tmp_err = (werror) calloc(1, sizeof(struct _w_error));

	for (; x < (nneur-1); x++){
		register wneuron neuron_tmp = neurons[nneur-x-1];
		ret = neuron_tmp->backward(neuron_tmp, error[x], cache[nneur-x-1], &error[x+1]);

		if (ret != CL_SUCCESS) break;
	}

	if (ret != CL_SUCCESS) {
		for (uint32_t y=1; y < x; y++) wekuaFreeMatrix(error[x], 0, NULL);
	}

	ret = neuron_tmp->backward(neuron_tmp, error[nneur-1], cache[nneur-1], err);

	return ret;
}
#include "../headers/network.h"

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
	uint8_t d = 0;
	uint32_t nneur = net->nneur;
	wneuron *neurons = net->neurons;

	if (cache != NULL){
		cache[0] = (wcache*) calloc(nneur, sizeof(wcache));

		tmp[d] = runWekuaNeuron(neurons[0], input, &cache[0][0], 0, NULL);
		for (uint32_t x=1; x<nneur; x++){
			d ^= 1;
			if ((tmp[d] = runWekuaNeuron(neurons[x], tmp[d^1], &cache[0][x], 0, NULL)) == NULL) break;
		}
	}else{
		tmp[d] = runWekuaNeuron(neurons[0], input, NULL, 0, NULL);
		for (uint32_t x=1; x<nneur; x++){
			d ^= 1;
			if ((tmp[d] = runWekuaNeuron(neurons[x], tmp[d^1], NULL, 0, NULL)) == NULL) break;
			wekuaFreeMatrix(tmp[d^1], 0, NULL);
		}
	}

	return tmp[d];
}

int wekuaNetworkBackward(wnetwork net, werror *error, wcache *cache, werror *err){
	if (net == NULL || error == NULL || cache == NULL) return CL_INVALID_ARG_VALUE;

	int ret = CL_SUCCESS;
	uint32_t nneur = net->nneur;
	uint32_t x = 0;
	wneuron *neurons = net->neurons;

	for (; x < (nneur-1); x++){
		ret = wekuaNeuronBackward(neurons[nneur-1-x], error[x], cache[nneur-1-x], &error[x+1]);
		if (ret != CL_SUCCESS) break;
	}

	if (ret != CL_SUCCESS) {
		for (uint32_t y=1; y < x; y++) wekuaFreeNeuronError(error[y]);
	}else{
		ret = wekuaNeuronBackward(neurons[0], error[nneur-1], cache[0], err);
	}

	return ret;
}

void wekuaFreeNeuron(wneuron neur, uint32_t nw, cl_event *be){
	if (neur == NULL) return;
	clWaitForEvents(nw, be);

	uint64_t layers = neur->layer;
	wmatrix *w, *b;
	w = neur->weight; b = neur->bias;
	for (uint64_t i = 0; i < layers; i++){
		wekuaFreeMatrix(w[i], 0, NULL);
		if (b != NULL) wekuaFreeMatrix(b[i], 0, NULL);
	}
	free(neur);
}

void wekuaFreeNetwork(wnetwork net, uint32_t nw, cl_event *be){
	if (net == NULL) return;
	clWaitForEvents(nw, be);
	uint32_t nneur = net->nneur;
	wneuron *neurons = net->neurons;
	for (uint32_t i = 0; i < nneur; i++) wekuaFreeNeuron(neurons[i], 0, NULL);
	free(net);
}

void wekuaFreeNetCache(wnetwork net, wcache *cache){
	if (net == NULL || cache == NULL) return;

	uint32_t nneur = net->nneur;
	for (uint32_t x=0; x<nneur; x++) wekuaFreeNeuronCache(cache[x]);
	free(cache);
}

void wekuaFreeNetError(wnetwork net, werror *error){
	if (net == NULL || error == NULL) return;

	uint32_t nneur = net->nneur;
	for (uint32_t x=0; x<nneur; x++) wekuaFreeNeuronError(error[x]);
}
#include "wekua.h"

wmatrix runWekuaNetwork(wnetwork net, wmatrix input, wcache **cache);
void backward_net(wneuron *neurons, uint32_t nneur, werror error, wcache *cache, werror *err);

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

			if (tmp[d] == NULL) break;

			d ^= 1;
			if (tmp[d] != input) wekuaFreeMatrix(tmp[d], 0, NULL);
		}
	}else{
		tmp[0] = input;
		for (uint32_t x=0; x<nneur; x++){
			tmp[d] = neurons[x]->run(neurons[x], tmp[d^1], NULL, 0, NULL);

			if (tmp[d] == NULL) break;

			d ^= 1;
			if (tmp[d] != input) wekuaFreeMatrix(tmp[d], 0, NULL);
		}
	}

	wekuaFreeMatrix(tmp[d^1], 0, NULL);

	return output;
}
#ifndef NETWORK_H
#define NETWORK_H

#include "neuron.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _w_net {
	wneuron *neurons;
	uint32_t nneur;
	uint8_t dtype;
} *wnetwork;

wnetwork wekuaNeuronNetwork(uint32_t neur_num, uint8_t dtype);
void *runWekuaNetwork(wnetwork net, void *input, wcache **cache);
int wekuaNetworkBackward(wnetwork net, werror *error, wcache *cache, werror *err);

uint8_t saveWekuaNetwork(const char *name, wnetwork net);
uint8_t loadWekuaNetwork(const char *name, wnetwork net, wekuaContext ctx);

void wekuaFreeNetCache(wnetwork net, wcache *cache);
void wekuaFreeNetError(wnetwork net, werror *error);
void wekuaFreeNetwork(wnetwork net, uint32_t nw, cl_event *be);

#ifdef __cplusplus
}
#endif
#endif
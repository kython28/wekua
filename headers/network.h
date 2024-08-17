#ifndef WEKUA_NETWORK_H
#define WEKUA_NETWORK_H

#include "neuron.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WEKUA_NONE_REGULARIZATION 0
#define WEKUA_L1_REGULARIZATION 1
#define WEKUA_L2_REGULARIZATION 2

typedef struct _w_net {
	wneuron *neurons;
	uint32_t nneur;
	uint8_t dtype;
} *wnetwork;

wnetwork wekuaNeuronNetwork(uint32_t neur_num, uint8_t dtype);
wmatrix runWekuaNetwork(wnetwork net, wmatrix input, wcache **cache);
int wekuaNetworkBackward(
	wnetwork net, werror *error, wcache *cache, werror *err,
	void *alpha, void *beta, uint8_t regularization_type
) __attribute__ ((warn_unused_result));

uint8_t saveWekuaNetwork(const char *name, wnetwork net) __attribute__ ((warn_unused_result));
uint8_t loadWekuaNetwork(const char *name, wnetwork net, wekuaContext ctx) __attribute__ ((warn_unused_result));

void wekuaFreeNetCache(wnetwork net, wcache *cache);
void wekuaFreeNetError(wnetwork net, werror *error);
void wekuaFreeNetwork(wnetwork net, uint32_t nw, cl_event *be);

#ifdef __cplusplus
}
#endif
#endif

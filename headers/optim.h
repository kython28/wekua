#ifndef WEKUA_OPTIM_H
#define WEKUA_OPTIM_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _w_optim {
	wekuaContext ctx;
	wnetwork net;
	void *params; // Data for the Optimizer
	void *others;

	uint8_t dtype;

	// To update the weight
	int (*step)(void *, void *, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix);
	int (*zero)(void *);
	void (*free_func)(void *optim, uint32_t nw, cl_event *be);
} *woptim;

woptim wekuaOptimGD(wekuaContext ctx, wnetwork net, void *lr, void *lri, uint8_t dtype); // Gradient descent
woptim wekuaOptimGDM(wekuaContext ctx, wnetwork net, void *lr, void *lri, void *beta, void *betai, uint8_t dtype); // Gradient Descent momentum 
woptim wekuaOptimNAG(wekuaContext ctx, wnetwork net, void *lr, void *lri, void *beta, void *betai, uint8_t dtype); // Nesterov Accelerated Gradient
woptim wekuaOptimAdaGrad(wekuaContext ctx, wnetwork net, void *lr, void *lri, uint8_t dtype); // Adaptive gradient optimizatione
woptim wekuaOptimRMSProp(wekuaContext ctx, wnetwork net, void *lr, void *lri, void *beta, void *betai, uint8_t dtype); // Root Mean Square Propagation
woptim wekuaOptimAdadelta(wekuaContext ctx, wnetwork net, void *lr, void *lri, uint8_t dtype); // Adadelta
woptim wekuaOptimAdam(wekuaContext ctx,  wnetwork net, void *lr, void *lri, void *beta1, void *beta1i, void *beta2, void *beta2i, uint8_t dtype); // Adam

int wekuaOptimStep(woptim optim, werror *error, wcache *cache) __attribute__ ((warn_unused_result));
int wekuaOptimZero(woptim optim) __attribute__ ((warn_unused_result));

void wekuaFreeOptim(woptim optim, uint32_t nw, cl_event *be);

#ifdef __cplusplus
}
#endif
#endif

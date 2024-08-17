#ifndef WEKUA_ACTI_H
#define WEKUA_ACTI_H

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _w_acti {
	void *data; // Activation function data
	int (*run_acti)(void *, wmatrix, uint32_t, cl_event *); // To run the activation function
	wmatrix (*get_dev)(void *, wmatrix); // To get derivate
	void (*free_func)(void *, uint32_t, cl_event *); // To free the wacti object
} *wacti;

wacti wekuaActiLinear(void); // -> x
wacti wekuaActiTanh(void); // -> wekuaMatrixTanh(x)
wacti wekuaActiSigmoid(void); // -> 1/(1 + e^(-x))
wacti wekuaActiReLU(void); // -> max(0, x)
wacti wekuaActiLeakyReLU(wekuaContext ctx, void *alpha, void *alphai, uint8_t dtype); // -> max(alpha*x, x) & (0.0 < alpha < 1.0)
// wacti wekuaActiELU(); // -> alpha*(e^x - 1)


int runWekuaActi(wacti acti, wmatrix input, uint32_t nw, cl_event *be) __attribute__ ((warn_unused_result));
wmatrix wekuaActiGetDev(wacti acti, wmatrix output);
void wekuaFreeActi(wacti acti, uint32_t nw, cl_event *be);

#ifdef __cplusplus
}
#endif
#endif

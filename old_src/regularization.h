#ifndef WEKUA_REGULARIZATION_H
#define WEKUA_REGULARIZATION_H

#include "../headers/matrix.h"

wmatrix wekuaL1Regularization(wmatrix weight, void *alpha, void *beta);
wmatrix wekuaL2Regularization(wmatrix weight, void *alpha, void *beta);
int wekuaAddRegularization(wmatrix regularization, wmatrix dev_error, uint32_t nw, cl_event *be, cl_event *e);

#endif


#include "wekua.h"

// Mean Absolute Error

void runMAE(wmatrix *x, wmatrix *y, double *real_error, double *imag_error, uint32_t nw, cl_event *be){
	if (x == NULL || y == NULL){
		return;
	}
	cl_event ie[2];

	wmatrix *z = wekuaMatrixCopy(x, nw, be, ie), *g;
	wekuaMatrixSub(z, y, 1, ie, &ie[1]);
	g = wekuaMatrixAbs(z, 1, &ie[1]);
	wekuaMatrixMean(g, real_error, imag_error, 0, NULL);
	wekuaFreeMatrix(z, 0, NULL);
	wekuaFreeMatrix(g, 0, NULL);

	clReleaseEvent(ie[0]);
	clReleaseEvent(ie[1]);
}

wmatrix *devMAE(wmatrix *x, wmatrix *y, uint32_t nw, cl_event *be){
	cl_event ie[3];

	wmatrix *er, *aer;
	er = wekuaMatrixCopy(y, nw, be, ie);
	wekuaMatrixSub(er, x, 1, ie, &ie[1]);
	aer = wekuaMatrixAbs(er, 1, &ie[1]);

	wekuaMatrixDivide(er, aer, 0, NULL, &ie[2]);
	wekuaFreeMatrix(aer, 1, &ie[2]);

	for (uint32_t j=0; j<3; j++) clReleaseEvent(ie[j]);

	return er;
}

wloss *wekuaMAE(){
	wloss *l = calloc(1, sizeof(wloss));
	l->func = &runMAE;
	l->get_dev = &devMAE;
	return l;
}


// Mean Square Error

void runMSE(wmatrix *x, wmatrix *y, double *real_error, double *imag_error, uint32_t nw, cl_event *be){
	cl_event e[3];

	wmatrix *z = wekuaMatrixCopy(x, nw, be, e);
	wekuaMatrixSub(z, y, 1, e, &e[1]);
	wekuaMatrixDot(z, z, 1, &e[1], &e[2]);
	wekuaMatrixMean(z, real_error, imag_error, 1, &e[2]);
	wekuaFreeMatrix(z, 0, NULL);

	for (uint32_t j=0; j<3; j++) clReleaseEvent(e[j]);
}

wmatrix *devMSE(wmatrix *x, wmatrix *y, uint32_t nw, cl_event *be){
	cl_event e[3];

	wmatrix *er = wekuaMatrixCopy(x, nw, be, e);
	wekuaMatrixSub(er, y, 1, e, &e[1]);
	wekuaMatrixDotScalar(er, -2.0, 0.0, 1, &e[1], &e[2]);

	clWaitForEvents(1, &e[2]);
	for (uint32_t j=0; j<3; j++) clReleaseEvent(e[j]);

	return er;
}

wloss *wekuaMSE(){
	wloss *l = calloc(1, sizeof(wloss));
	l->func = &runMSE;
	l->get_dev = &devMSE;
	return l;
}

void runWekuaLoss(wmatrix *output, wmatrix *ow, double *real, double *imag, wloss *l, uint32_t nw, cl_event *be){
	if (output == NULL || ow == NULL || l == NULL){
		return;
	}else if (real == imag){
		return;
	}
	l->func(ow, output, real, imag, nw, be);
}

void wekuaFreeLoss(wloss *l, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (l == NULL){
		return;
	}
	free(l);
}
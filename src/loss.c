#include "wekua.h"

// Mean Absolute Error

void runMAE(wmatrix *x, wmatrix *y, double *real_error, double *imag_error){
	if (x == NULL || y == NULL){
		return;
	}
	wmatrix *z = wekuaMatrixCopy(x);
	wekuaMatrixSub(z, y);
	wekuaMatrixAbs(z);
	wekuaMatrixMean(z, real_error, imag_error);
	wekuaFreeMatrix(z);
}

wmatrix *devMAE(wmatrix *x, wmatrix *y){
	wmatrix *er, *aer;
	er = wekuaMatrixCopy(y);
	wekuaMatrixSub(er, x);
	aer = wekuaMatrixCopy(er);
	wekuaMatrixAbs(aer);
	wekuaMatrixDivide(er, aer);
	wekuaFreeMatrix(aer);
	return er;
}

wloss *wekuaMAE(){
	wloss *l = calloc(1, sizeof(wloss));
	l->func = &runMAE;
	l->get_dev = &devMAE;
	return l;
}


// Mean Square Error

void runMSE(wmatrix *x, wmatrix *y, double *real_error, double *imag_error){
	if (x == NULL || y == NULL){
		return;
	}
	wmatrix *z = wekuaMatrixCopy(x);
	wekuaMatrixSub(z, y);
	wekuaMatrixDot(z, z);
	wekuaMatrixMean(z, real_error, imag_error);
	wekuaFreeMatrix(z);
}

wmatrix *devMSE(wmatrix *x, wmatrix *y){
	wmatrix *er = wekuaMatrixCopy(x);
	wekuaMatrixSub(er, y);
	wekuaMatrixDotScalar(er, -2.0, 0.0);
	return er;
}

wloss *wekuaMSE(){
	wloss *l = calloc(1, sizeof(wloss));
	l->func = &runMSE;
	l->get_dev = &devMSE;
	return l;
}


void wekuaFreeLoss(wloss *l){
	if (l == NULL){
		return;
	}
	free(l);
}
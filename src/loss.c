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
	er = wekuaMatrixCopy(x);
	wekuaMatrixSub(er, y);
	aer = wekuaMatrixCopy(er);
	wekuaMatrixAbs(aer);
	wekuaMatrixDivide(er, aer);
	wekuaMatrixDotScalar(er, -1.0, 0.0);
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

// Negative Log-Likelihood Loss

void runNLLLoss(wmatrix *x, wmatrix *y, double *real_error, double *imag_error){
	if (x == NULL || y == NULL){
		return;
	}
	wmatrix *z = wekuaMatrixCopy(y);
	wekuaMatrixLn(z);
	wekuaMatrixDotScalar(z, -1.0, 0.0);
	wekuaMatrixMean(z, real_error, imag_error);
	wekuaFreeMatrix(z);
}

wmatrix *devNLLLoss(wmatrix *x, wmatrix *y){
	wmatrix *er = wekuaFillMatrix(y->ctx, y->r, y->c, -1.0, 0.0);
	wekuaMatrixDivide(er, y);
	return er;
}

wloss *wekuaNLLLoss(){
	wloss *l = calloc(1, sizeof(wloss));
	l->func = &runNLLLoss;
	l->get_dev = &devNLLLoss;
	return l;
}

// Cross Entropy Loss

void runCrossEntropyLoss(wmatrix *x, wmatrix *y, double *real_error, double *imag_error){
	if (x == NULL || y == NULL){
		return;
	}
	wmatrix *z = wekuaMatrixCopy(y);
	wekuaMatrixLn(z);
	wekuaMatrixDot(z, x);
	wekuaMatrixDotScalar(z, -1.0, 0.0);
	wekuaMatrixMean(z, real_error, imag_error);
	wekuaFreeMatrix(z);
}

wmatrix *devCrossEntropyLoss(wmatrix *x, wmatrix *y){
	wmatrix *er = wekuaMatrixCopy(x);
	wekuaMatrixDotScalar(er, -1.0, 0.0);
	wekuaMatrixDivide(er, y);
	return er;
}

wloss *wekuaCrossEntropyLoss(){
	wloss *l = calloc(1, sizeof(wloss));
	l->func = &runCrossEntropyLoss;
	l->get_dev = &devCrossEntropyLoss;
	return l;
}
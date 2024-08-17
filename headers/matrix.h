#ifndef WEKUA_MATRIX_H
#define WEKUA_MATRIX_H

#include "wekua.h"
#include <string.h>

// Data types

#define WEKUA_DTYPE_INT8 0
#define WEKUA_DTYPE_UINT8 1

#define WEKUA_DTYPE_INT16 2
#define WEKUA_DTYPE_UINT16 3

#define WEKUA_DTYPE_INT32 4
#define WEKUA_DTYPE_UINT32 5

#define WEKUA_DTYPE_INT64 6
#define WEKUA_DTYPE_UINT64 7

#define WEKUA_DTYPE_FLOAT 8
#define WEKUA_DTYPE_DOUBLE 9

#ifdef __cplusplus
extern "C" {
#endif

// To get random buffer from /dev/urandom
int getRandomBuffer(void *buf, size_t size);

typedef struct _wk_matrix {
	// Free function
	int (*free)(void *);

	cl_mem real; // Real numbers
	cl_mem imag; // Imaginary numbers

	// void *raw_real; // Real numbers array mapped
	// void *raw_imag; // Imaginary numbers array mapped

	wekuaContext ctx;

	// Matrix shape
	uint64_t shape[2];
	uint64_t vl_shape[3];
	uint64_t length, col, row, size;

	// Data type
	uint8_t dtype;

	// Does the matrix use complex elements?
	uint8_t com;

	// Info for OpenCL
	uint64_t work_items[9];
} *wmatrix;

void wekuaMatrixPrint(wmatrix a, uint32_t nw, cl_event *be); // To print wmatrix
uint8_t createComplexMatrix(wmatrix a) __attribute__ ((warn_unused_result)); // To enable complex numbers.
int removeComplexMatrix(wmatrix b, uint32_t nw, cl_event *be); // To disable complex numbers.

void getLWI(const uint64_t *global_items, uint64_t *local_items, uint32_t si, uint64_t max);
int mem_set_zero(wmatrix a, cl_mem buf);

wmatrix wekuaMatrixEmpty(wekuaContext ctx, uint64_t r, uint64_t c, uint8_t dtype); // To alloc an empty matrix
wmatrix wekuaAllocMatrix(wekuaContext ctx, uint64_t r, uint64_t c, uint8_t dtype); // To alloc a null matrix
wmatrix wekuaAllocComplexMatrix(wekuaContext ctx, uint64_t r, uint64_t c, uint8_t dtype); // To Alloc an empty matrix with complex elements
wmatrix wekuaFillMatrix(wekuaContext ctx, uint64_t r, uint64_t c, void *alpha, void *beta, uint8_t dtype); // To get matrix filled with same elements. Alpha is real number and Beta is imaginary number
wmatrix wekuaMatrixRandn(wekuaContext ctx, uint64_t r, uint64_t c, uint8_t com); // To get matrix with random elements
wmatrix wekuaMatrixRandUniform(wekuaContext ctx, uint64_t r, uint64_t c, void *ra, void *ia, void *re, void *ie, uint8_t dtype); // To get matrix with random numbers in the range [a, b) or [a, b] depending on rounding.
wmatrix wekuaMatrixCopy(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);
wmatrix wekuaMatrixResize(wmatrix a, uint64_t r, uint64_t c, void *alpha, void *beta); // To resize a matrix
wmatrix wekuaMatrixConvert(wmatrix a, uint8_t dtype, uint32_t nw, cl_event *be, cl_event *e);
wmatrix wekuaMatrixFromBuffer(wekuaContext ctx, uint64_t r, uint64_t c, void *rbuf, void *ibuf, uint8_t dtype);

int wekuaMatrixCopyBuffer(wmatrix a, void *rbuf, void *ibuf) __attribute__ ((warn_unused_result));
int wekuaMatrixWritetoBuffer(wmatrix a, void *rbuf, void *ibuf) __attribute__ ((warn_unused_result));

void wekuaGetValueFromMatrix(wmatrix a, uint64_t y, uint64_t x, void *real, void *imag, uint32_t nw, cl_event *be);
void wekuaPutValueToMatrix(wmatrix a, uint64_t y, uint64_t x, void *real, void *imag, uint32_t nw, cl_event *be);

int wekuaCopyMatrixRegion(
	wmatrix src, uint64_t src_offset_x, uint64_t src_offset_y,
	wmatrix dst, uint64_t dst_offset_x, uint64_t dst_offset_y,
	uint64_t w, uint64_t h
) __attribute__ ((warn_unused_result));

// Some BLAS functions
int wekuaBlasAxpy(wmatrix x, wmatrix y, void *alpha, void *beta, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // y = (alpha+beta*j)*x + y
int wekuaBlasScalar(wmatrix x, void *alpha, void *beta, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // Dot all elements in a matrix for a scalar. Alpha is real number and Beta is imaginary number
int wekuaBlasGemm(void *ralpha, void *ialpha, uint8_t a_trans, wmatrix a, uint8_t b_trans, wmatrix b,
	void *rbeta, void *ibeta, wmatrix c, uint32_t nw, cl_event *be
) __attribute__ ((warn_unused_result));
 
// Basic functions
wmatrix wekuaMatrixIden(wekuaContext ctx, uint64_t col, uint8_t dtype);
wmatrix wekuaMatrixTrans(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);
wmatrix wekuaMatrixDiag(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);
wmatrix wekuaMatrixAbs(wmatrix a, uint32_t nw, cl_event *be);
wmatrix wekuaMatrixAbsdiff(wmatrix a, wmatrix b, uint32_t nw, cl_event *be);
wmatrix wekuaMatrixArange(wekuaContext ctx,
	double start_r, double start_i, double end_r, double end_i,
	double delta, uint8_t trans
);

int wekuaMatrixAdd(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // Matrix addition
int wekuaMatrixAddScalar(wmatrix a, void *alpha, void *beta, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // Add to every values a scalar -> a_{i} += alpha + beta*j
int wekuaMatrixSub(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // Matrix Substration
int wekuaMatrixDot(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // Hadamard product
int wekuaMatrixDivide(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // a_{i} /= b_{i}
int wekuaMatrixPower(wmatrix a, wmatrix b, void *exp_r, void *exp_i, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // a_{i} = a_{i}^{exp_r+exp_i*j} or a_{i} = a_{i}^{b_{i}}
int wekuaMatrixLn(wmatrix a, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // a_{i} = ln(a_{i})
int wekuaMatrixLog(wmatrix a, wmatrix b, void *base_r, void *base_i) __attribute__ ((warn_unused_result));
int wekuaMatrixTrace(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be) __attribute__ ((warn_unused_result)); // Matrix Trace
int wekuaMatrixSqrt(wmatrix a, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result)); // a_{i} = sqrt(a_{i})

// Linear functions
wmatrix wekuaMatrixInv(wmatrix a, uint32_t nw, cl_event *be); // Matrix inverse
wmatrix wekuaMatrixSolve(wmatrix a, wmatrix b, uint32_t nw, cl_event *be);
wmatrix wekuaMatrixPinv(wmatrix a, uint32_t nw, cl_event *be); // Matrix Pseudo-inverse
uint64_t wekuaMatrixRang(wmatrix a, uint32_t nw, cl_event *be); // Matrix range
int wekuaMatrixDet(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be) __attribute__ ((warn_unused_result)); // Matrix determinant

// Trigonometric functions
int wekuaMatrixSin(wmatrix a, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result));
int wekuaMatrixCos(wmatrix a, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result));
int wekuaMatrixTan(wmatrix a, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result));
int wekuaMatrixSinh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result));
int wekuaMatrixCosh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result));
int wekuaMatrixTanh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e) __attribute__ ((warn_unused_result));

// Extra functions
int wekuaMatrixSum(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be) __attribute__ ((warn_unused_result)); // Sum of all the elements
int wekuaMatrixMul(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be) __attribute__ ((warn_unused_result)); // Mul of all the elements
int wekuaMatrixMean(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be) __attribute__ ((warn_unused_result)); // Mean of all the elements
void wekuaMatrixMax(wmatrix a, uint64_t *y, uint64_t *x, uint32_t nw, cl_event *be); // To get max value.
void wekuaMatrixMin(wmatrix a, uint64_t *y, uint64_t *x, uint32_t nw, cl_event *be); // To get min value.
wmatrix wekuaMatrixPoly(wmatrix a, uint32_t nw, cl_event *be); // Matrix Poly (Leverrier)
wmatrix wekuaMatrixEulerIden(wmatrix angle, uint32_t nw, cl_event *be);
wmatrix wekuaMatrixRoot(wmatrix a, uint32_t nw, cl_event *be); // Polynomial roots

void wekuaFreeMatrix(wmatrix a, uint32_t nw, cl_event *be); // To free a matrix

uint8_t saveWekuaMatrix(const char *name, wmatrix a);
wmatrix loadWekuaMatrix(const char *name, wekuaContext ctx);

#ifdef __cplusplus
}
#endif
#endif


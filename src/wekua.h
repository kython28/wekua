#ifndef WEKUA_H
#define WEKUA_H

#include <CL/cl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define WEKUA_DEVICE_TYPE_CPU CL_DEVICE_TYPE_CPU
#define WEKUA_DEVICE_TYPE_GPU CL_DEVICE_TYPE_GPU
#define WEKUA_DEVICE_TYPE_ALL CL_DEVICE_TYPE_ALL
#define WEKUA_DEVICE_TYPE_ACCELERATOR CL_DEVICE_TYPE_ACCELERATOR
#define WEKUA_DEVICE_TYPE_CUSTOM CL_DEVICE_TYPE_CUSTOM
#define WEKUA_DEVICE_TYPE_DEFAULT CL_DEVICE_TYPE_DEFAULT

typedef cl_device_type wekua_device_type;

typedef struct {
	cl_platform_id platform; // Platform ID
	uint8_t *name; // Platform name
	uint64_t nlen; // Devices numbers
} wPlatform;

typedef struct {
	cl_device_id device; // Device ID
	cl_device_type type; // Device type
	uint8_t *name; // Device name
	// Device Info
	uint32_t compute_units, clock_frequency, max_work_item_dimensions;
	uint64_t max_work_group_size, *max_work_item_sizes, nlen, max_size;
} wDevice;

uint32_t getPlatforms(wPlatform **platform);
uint32_t getDevices(wPlatform platform , wDevice **device, wekua_device_type type);
void freeWekuaPlatform(wPlatform *plat, uint32_t nplat);
void freeWekuaDevice(wDevice *dev, uint32_t ndev);

typedef struct {
	cl_context ctx; // OpenCL Context
	cl_command_queue command_queue; // OpenCL Command Queue
	cl_program *programs; // OpenCL programs
	cl_kernel *kernels; // OpenCL kernels
	// Info
	uint64_t max_work_item_dimensions;
	uint64_t max_work_group_size, *max_work_item_sizes;
} wekuaContext;

wekuaContext *createWekuaContext(wDevice *dev);
wekuaContext *createSomeWekuaContext(wekua_device_type type);
void freeWekuaContext(wekuaContext *context);

void getRandomBuffer(void *buf, uint64_t size);

// Wekua Matrix
typedef struct {
	cl_mem real; // Real numbers
	cl_mem imag; // Imaginary numbers

	double *raw_real; // Real numbers array mapped
	double *raw_imag; // Imaginary numbers array mapped

	wekuaContext *ctx;

	uint32_t r,c; // Dimensions

	// Does the matrix use complex elements?
	uint8_t com;

	// Info for OpenCL
	uint64_t size, work_items[3];
} wmatrix;

void wekuaMatrixPrint(wmatrix *a);
uint8_t createComplexMatrix(wmatrix *a);
void removeComplexMatrix(wmatrix *a);

wmatrix *wekuaAllocMatrix(wekuaContext *ctx, uint32_t r, uint32_t c); // To alloc an empty matrix
wmatrix *wekuaAllocComplexMatrix(wekuaContext *ctx, uint32_t r, uint32_t c); // To Alloc an empty matrix with complex elements
wmatrix *wekuaFillMatrix(wekuaContext *ctx, uint32_t r, uint32_t c, double alpha, double beta); // To get matrix filled with same elements. Alpha is real number and Beta is imaginary number
wmatrix *wekuaMatrixRandn(wekuaContext *ctx, uint32_t r, uint32_t c, uint8_t com); // To get matrix with random elements
wmatrix *wekuaMatrixRandUniform(wekuaContext *ctx, uint32_t r, uint32_t c, double ra, double ia, double re, double ie, uint8_t com); // To get matrix with random numbers in the range [a, b) or [a, b] depending on rounding.
wmatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint32_t r, uint32_t c, void *rbuf, void *ibuf); // To create Matrix from buffer
wmatrix *wekuaMatrixCopy(wmatrix *a); // To copy a matrix
wmatrix *wekuaCutMatrix(wmatrix *a, uint32_t x, uint32_t w, uint32_t y, uint32_t h); // To get a submatrix
wmatrix *wekuaMatrixResize(wmatrix *a, uint32_t r, uint32_t c, double alpha, double beta); // To resize a matrix

// Basic functions
wmatrix *wekuaMatrixIden(wekuaContext *ctx, uint32_t c); // Identity Matrix
wmatrix *wekuaMatrixTrans(wmatrix *a); // Matrix Transpose
wmatrix *wekuaMatrixProduct(wmatrix *a, wmatrix *b); // Matrix Product
wmatrix *wekuaMatrixDiag(wmatrix *a);
void wekuaMatrixAdd(wmatrix *a, wmatrix *b); // Matrix addition
void wekuaMatrixSub(wmatrix *a, wmatrix *b); // Matrix Substration
void wekuaMatrixDot(wmatrix *a, double alpha, double beta); // Dot all elements in a matrix for a number. Alpha is real number and Beta is imaginary number
void wekuaMatrixAbs(wmatrix *a);
void wekuaMatrixAbsdiff(wmatrix *a, wmatrix *b);

// Trigonometric functions
void wekuaMatrixSin(wmatrix *a);
void wekuaMatrixCos(wmatrix *a);
void wekuaMatrixTan(wmatrix *a);
void wekuaMatrixSinh(wmatrix *a);
void wekuaMatrixCosh(wmatrix *a);
void wekuaMatrixTanh(wmatrix *a);

// Extra functions
void wekuaMatrixSum(wmatrix *a, double *real, double *imag); // Sum of all the elements
void wekuaMatrixMul(wmatrix *a, double *real, double *imag); // Mul of all the elements
void wekuaMatrixMean(wmatrix *a, double *real, double *imag); // Mean of all the elements
void wekuaMatrixNorm(wmatrix *a, double *real, double *imag); // Matrix Norm
void wekuaMatrixTrace(wmatrix *a, double *real, double *imag); // Matrix Trace
void wekuaMatrixToComplex(wmatrix *a, double *real, double *imag); // Matrix to Complex number
void wekuaMatrixMax(wmatrix *a, double *real, double *imag); // To get max value.
void wekuaMatrixMin(wmatrix *a, double *real, double *imag); // To get min value.
wmatrix *wekuaComplexToMatrix(wekuaContext *ctx, double r, double i); // Complex number to Matrix
wmatrix *wekuaComplexRandomToMatrix(wekuaContext *ctx); // Random Complex number to Matrix
wmatrix *wekuaMatrixPoly(wmatrix *a); // Matrix Poly (Leverrier)
wmatrix *wekuaMatrixRoot(wmatrix *a); // Polynomial roots
wmatrix *wekuaMatrixPower(wmatrix *a, int64_t n); // Matrix power

// Linalg functions
void wekuaMatrixDet(wmatrix *a, double *real, double *imag); // Matrix Determinant
wmatrix *wekuaMatrixInv(wmatrix *a); // Matrix Inverse
wmatrix *wekuaMatrixSolve(wmatrix *a, wmatrix *b);
wmatrix *wekuaMatrixPinv(wmatrix *a); // Matrix Pseudoinverse
uint32_t wekuaMatrixRang(wmatrix *a); // Matrix Rang
wmatrix *wekuaMatrixEigenvalues(wmatrix *a); // Eigenvalues

void wekuaFreeMatrix(wmatrix *a); // To free a matrix


#endif

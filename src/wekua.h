#ifndef WEKUA_H
#define WEKUA_H

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

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

// To get random buffer from /dev/urandom
void getRandomBuffer(void *buf, uint64_t size);

// Wekua Matrix
typedef struct {
	cl_mem real; // Real numbers
	cl_mem imag; // Imaginary numbers

	double *raw_real; // Real numbers array mapped
	double *raw_imag; // Imaginary numbers array mapped

	wekuaContext *ctx;

	uint64_t shape[2], size; // Dimensions

	// Does the matrix use complex elements?
	uint8_t com;

	// Info for OpenCL
	uint64_t work_items[5];
} wmatrix;

void wekuaMatrixPrint(wmatrix *a); // To print wmatrix
uint8_t createComplexMatrix(wmatrix *a); // To enable complex numbers.
void removeComplexMatrix(wmatrix *a); // To disable complex numbers.

wmatrix *wekuaAllocMatrix(wekuaContext *ctx, uint64_t r, uint64_t c); // To alloc an empty matrix
wmatrix *wekuaAllocComplexMatrix(wekuaContext *ctx, uint64_t r, uint64_t c); // To Alloc an empty matrix with complex elements
wmatrix *wekuaFillMatrix(wekuaContext *ctx, uint64_t r, uint64_t c, double alpha, double beta); // To get matrix filled with same elements. Alpha is real number and Beta is imaginary number
wmatrix *wekuaMatrixRandn(wekuaContext *ctx, uint64_t r, uint64_t c, uint8_t com); // To get matrix with random elements
wmatrix *wekuaMatrixRandUniform(wekuaContext *ctx, uint64_t r, uint64_t c, double ra, double ia, double re, double ie, uint8_t com); // To get matrix with random numbers in the range [a, b) or [a, b] depending on rounding.
wmatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint64_t r, uint64_t c, void *rbuf, void *ibuf); // To create Matrix from buffer
wmatrix *wekuaMatrixCopy(wmatrix *a); // To copy a matrix
wmatrix *wekuaCutMatrix(wmatrix *a, uint64_t x, uint64_t w, uint64_t y, uint64_t h); // To get a submatrix
wmatrix *wekuaMatrixResize(wmatrix *a, uint64_t r, uint64_t c, double alpha, double beta); // To resize a matrix

// Basic functions
wmatrix *wekuaMatrixIden(wekuaContext *ctx, uint64_t c); // Identity Matrix
wmatrix *wekuaMatrixTrans(wmatrix *a); // Matrix Transpose
wmatrix *wekuaMatrixProduct(wmatrix *a, wmatrix *b); // Matrix Product
wmatrix *wekuaMatrixDiag(wmatrix *a);
wmatrix *wekuaArange(wekuaContext *ctx, double x, double y, double alpha);
void wekuaMatrixAdd(wmatrix *a, wmatrix *b); // Matrix addition
void wekuaMatrixSub(wmatrix *a, wmatrix *b); // Matrix Substration
void wekuaMatrixDotScalar(wmatrix *a, double alpha, double beta); // Dot all elements in a matrix for a scalar. Alpha is real number and Beta is imaginary number
void wekuaMatrixDot(wmatrix *a, wmatrix *b);
void wekuaMatrixAbs(wmatrix *a);
void wekuaMatrixAbsdiff(wmatrix *a, wmatrix *b);
void wekuaMatrixLn(wmatrix *a);
void wekuaMatrixLog(wmatrix *a, double r_base, double i_base);
void wekuaMatrixDivide(wmatrix *a, wmatrix *b);
void wekuaMatrixPowr(wmatrix *a, double real, double imag);

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

// I've not look at this, if you want to help me, let me know :-)
// wmatrix *wekuaMatrixEigenvalues(wmatrix *a); // Eigenvalues
// wmatrix *wekuaMatrixEigenvectors(wmatrix *a); // Eigenvectors
// wmatrix **wekuaMatrixEig(wmatrix *a); // Eigenvalues and EigenVectors

void wekuaFreeMatrix(wmatrix *a); // To free a matrix

// Wekua Network Module
typedef struct {
	void **data; // Module data
	wmatrix **cache; // Cache
	void (*acti_func)(wmatrix *); // Activation function
	wmatrix *(*func)(void *, wmatrix *); // Module function
	// void (*get_data)(void *module); // To get module data
	void (*set_cache_id)(void *, int64_t, void *, void *, uint32_t *, uint8_t *); // To set position in cache
	void (*free_func)(void *); // Free function
	uint8_t com;
	uint32_t nmod, *pseq;
	int64_t arch_id; // Position of the output into architecture cache
} wmodule;

// Wekua Network Architecture
typedef struct {
	wmodule **modules; // Modules
	uint32_t nmodule[3]; // Modules number
	wmatrix **weight;
	wmatrix **cache, **s; // Cache
	wmatrix *(*func)(wmodule **, uint32_t, wmatrix *); // Arch function
	uint32_t pseq;
	uint8_t com, *acti_func_id;
} warch;

// Wekua Loss
typedef struct {
	void (*func)(wmatrix *, wmatrix *, double*, double*);
	wmatrix *(*get_dev)(wmatrix *, wmatrix *);
} wloss;

warch *wekuaArch(wekuaContext *ctx, uint32_t nmodule, wmatrix *(*func)(wmodule **, uint32_t, wmatrix *), uint8_t com);
void addModuleToArch(warch *arch, wmodule *module);
void configureWekuaArch(warch *arch);
wmatrix *runWekuaArch(warch *arch, wmatrix *input);
void wekuaFreeArch(warch *arch);

// Activations functions
void wekuaSigmoid(wmatrix *a);
void wekuaTanh(wmatrix *a);
void wekuaReLU(wmatrix *a);
void wekuaLeakyReLU(wmatrix *a);
void wekuaSoftplus(wmatrix *a);

// Modules
wmodule *wekuaLinear(wekuaContext *ctx, uint64_t input, uint64_t output, uint32_t deep, void (*acti_func)(wmatrix *), uint8_t com);
wmatrix *runWekuaLinear(void *m, wmatrix *input);
void freeWekuaLinear(void *m);

wmodule *wekuaSequential(wekuaContext *ctx, uint32_t nmodule, uint8_t com);
void addModuleToSequential(wmodule *sequential, wmodule *module);
wmatrix *runWekuaSequential(void *m, wmatrix *input);
void freeWekuaSequential(void *m);

// Loss functions
wloss *wekuaMAE();
wloss *wekuaMSE();
wloss *wekuaNLLLoss();
wloss *wekuaCrossEntropyLoss();

// Optimization functions
void wekuaGradientDescent(double lr, warch *a, wmatrix *output, wmatrix *ow, wloss *l, double *real, double *imag);

#endif
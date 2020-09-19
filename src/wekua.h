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
#include <unistd.h>
#include <math.h>

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
	uint64_t max_work_group_size, *max_work_item_sizes, nlen, max_global_size, max_local_size;
} wDevice;

uint32_t getPlatforms(wPlatform **platform);
uint32_t getDevices(wPlatform platform , wDevice **device, cl_device_type type);
void freeWekuaPlatform(wPlatform *plat, uint32_t nplat);
void freeWekuaDevice(wDevice *dev, uint32_t ndev);

typedef struct {
	cl_context ctx; // OpenCL Context
	cl_command_queue command_queue; // OpenCL Command Queue
	cl_program *programs; // OpenCL programs
	cl_kernel *kernels; // OpenCL kernels
	// Info
	uint32_t compute_units;
	uint64_t max_work_group_size;
} wekuaContext;

wekuaContext *createWekuaContext(wDevice *dev);
wekuaContext *createSomeWekuaContext(cl_device_type type);
void freeWekuaContext(wekuaContext *context);

// To get random buffer from /dev/urandom
void getRandomBuffer(void *buf, uint64_t size);

// Wekua Matrix
typedef struct {
	void *parent;

	cl_mem real; // Real numbers
	cl_mem imag; // Imaginary numbers

	double *raw_real; // Real numbers array mapped
	double *raw_imag; // Imaginary numbers array mapped

	wekuaContext *ctx;

	uint64_t shape[2], offset[2]; // Dimensions
	uint64_t real_size[2], size; // Real dimension

	// Does the matrix use complex elements?
	uint8_t com;

	// is it a sub-matrix?
	uint8_t sm;

	// Info for OpenCL
	uint64_t work_items[5];
} wmatrix;

void wekuaMatrixPrint(wmatrix *a, uint32_t nw, cl_event *be); // To print wmatrix
uint8_t createComplexMatrix(wmatrix *a); // To enable complex numbers.
void removeComplexMatrix(wmatrix *b, uint32_t nw, cl_event *be); // To disable complex numbers.

wmatrix *wekuaAllocMatrix(wekuaContext *ctx, uint64_t r, uint64_t c); // To alloc an empty matrix
wmatrix *wekuaAllocComplexMatrix(wekuaContext *ctx, uint64_t r, uint64_t c); // To Alloc an empty matrix with complex elements
wmatrix *wekuaFillMatrix(wekuaContext *ctx, uint64_t r, uint64_t c, double alpha, double beta); // To get matrix filled with same elements. Alpha is real number and Beta is imaginary number
wmatrix *wekuaMatrixRandn(wekuaContext *ctx, uint64_t r, uint64_t c, uint8_t com); // To get matrix with random elements
wmatrix *wekuaMatrixRandUniform(wekuaContext *ctx, uint64_t r, uint64_t c, double ra, double ia, double re, double ie, uint8_t com); // To get matrix with random numbers in the range [a, b) or [a, b] depending on rounding.
wmatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint64_t r, uint64_t c, void *rbuf, void *ibuf); // To create Matrix from buffer
wmatrix *wekuaMatrixCopy(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e); // To copy a matrix
wmatrix *wekuaCutMatrix(wmatrix *a, uint64_t x, uint64_t w, uint64_t y, uint64_t h); // To get a submatrix
wmatrix *wekuaMatrixResize(wmatrix *a, uint64_t r, uint64_t c, double alpha, double beta, uint32_t nw, cl_event *be, cl_event *e); // To resize a matrix

void wekuaReshapeMatrix(wmatrix *a, uint64_t r, uint64_t c, uint32_t nw, cl_event *be);
void wekuaGetValueFromMatrix(wmatrix *a, uint64_t y, uint64_t x, double *real, double *imag, uint32_t nw, cl_event *be);
void wekuaPutValueToMatrix(wmatrix *a, uint64_t y, uint64_t x, double real, double imag, uint32_t nw, cl_event *be);

// Some BLAS functions
void wekuaBlasAxpy(double alpha, double beta, wmatrix *x, wmatrix *y, uint32_t nw, cl_event *be, cl_event *e); // y = (alpha+beta*j)*x + y
void wekuaBlasAsum(wmatrix *x, double *real, double *imag, uint32_t nw, cl_event *be);
void wekuaBlasNorm(wmatrix *x, double *real, double *imag, uint32_t nw, cl_event *be);

void wekuaBlasGemm(double ralpha, double ialpha, uint8_t a_trans, wmatrix *a, uint8_t b_trans, wmatrix *b,
	double rbeta, double ibeta, wmatrix *c, uint32_t nw, cl_event *be, cl_event *e
);

// Basic functions
wmatrix *wekuaMatrixIden(wekuaContext *ctx, uint64_t c); // Identity Matrix
wmatrix *wekuaMatrixTrans(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e); // Matrix Transpose
wmatrix *wekuaMatrixProduct(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e); // Matrix Product
wmatrix *wekuaMatrixDiag(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e);
wmatrix *wekuaArange(wekuaContext *ctx, double x, double y, double alpha);
wmatrix *wekuaMatrixAbs(wmatrix *a, uint32_t nw, cl_event *be);
void wekuaMatrixAdd(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e); // Matrix addition
void wekuaMatrixSub(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e); // Matrix Substration
void wekuaMatrixDotScalar(wmatrix *a, double alpha, double beta, uint32_t nw, cl_event *be, cl_event *e); // Dot all elements in a matrix for a scalar. Alpha is real number and Beta is imaginary number
void wekuaMatrixDot(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e); // Hadamard product
void wekuaMatrixAbsdiff(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be);
void wekuaMatrixLn(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e);
void wekuaMatrixLog(wmatrix *a, double r_base, double i_base, uint32_t nw, cl_event *be);
void wekuaMatrixDivide(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be, cl_event *e);
void wekuaMatrixPowr(wmatrix *a, double real, double imag, uint32_t nw, cl_event *be, cl_event *e);

// Trigonometric functions
void wekuaMatrixSin(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e);
void wekuaMatrixCos(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e);
void wekuaMatrixTan(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e);
void wekuaMatrixSinh(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e);
void wekuaMatrixCosh(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e);
void wekuaMatrixTanh(wmatrix *a, uint32_t nw, cl_event *be, cl_event *e);

// Extra functions
void wekuaMatrixSum(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be); // Sum of all the elements
void wekuaMatrixMul(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be); // Mul of all the elements
void wekuaMatrixMean(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be); // Mean of all the elements
void wekuaMatrixTrace(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be); // Matrix Trace
void wekuaMatrixToComplex(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be); // Matrix to Complex number
void wekuaMatrixMax(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be); // To get max value.
void wekuaMatrixMin(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be); // To get min value.
wmatrix *wekuaComplexToMatrix(wekuaContext *ctx, double r, double i); // Complex number to Matrix
wmatrix *wekuaComplexRandomToMatrix(wekuaContext *ctx); // Random Complex number to Matrix
wmatrix *wekuaMatrixPoly(wmatrix *a); // Matrix Poly (Leverrier)
wmatrix *wekuaMatrixRoot(wmatrix *a); // Polynomial roots
wmatrix *wekuaMatrixPower(wmatrix *a, int64_t n); // Matrix power

// Linalg functions
void wekuaMatrixDet(wmatrix *a, double *real, double *imag, uint32_t nw, cl_event *be); // Matrix Determinant
wmatrix *wekuaMatrixInv(wmatrix *a, uint32_t nw, cl_event *be); // Matrix Inverse
wmatrix *wekuaMatrixSolve(wmatrix *a, wmatrix *b, uint32_t nw, cl_event *be);
wmatrix *wekuaMatrixPinv(wmatrix *a, uint32_t nw, cl_event *be); // Matrix Pseudoinverse
uint32_t wekuaMatrixRang(wmatrix *a, uint32_t nw, cl_event *be); // Matrix Rang

// I've not look at this, if you want to help me, let me know :-)
// wmatrix *wekuaMatrixEigenvalues(wmatrix *a); // Eigenvalues
// wmatrix *wekuaMatrixEigenvectors(wmatrix *a); // Eigenvectors
// wmatrix **wekuaMatrixEig(wmatrix *a); // Eigenvalues and EigenVectors


int saveWekuaMatrix(const char *name, wmatrix *a);
wmatrix *openWekuaMatrix(wekuaContext *ctx, const char *name);
void wekuaFreeMatrix(wmatrix *a, uint32_t nw, cl_event *be); // To free a matrix

// Wekua Loss
typedef struct {
	void (*func)(wmatrix *, wmatrix *, double*, double*, uint32_t, cl_event *);
	wmatrix *(*get_dev)(wmatrix *, wmatrix *, uint32_t, cl_event *);
} wloss;


wloss *wekuaMAE();
wloss *wekuaMSE();
// wloss *wekuaCrossEntropyLoss();

void wekuaFreeLoss(wloss *l, uint32_t nw, cl_event *be);


// Activations functions
typedef struct {
	void *data;
	void (*acti)(void *, wmatrix *, uint32_t, cl_event *);
	wmatrix *(*acti_dev)(void *, wmatrix *, uint32_t, cl_event *);
} wacti;

wacti *wekuaFLinear();
wacti *wekuaSigmoid();
wacti *wekuaTanh();
wacti *wekuaReLU();
wacti *wekuaLeakyReLU(double alpha);

void runWekuaActi(wacti *a, wmatrix *b, uint32_t nw, cl_event *be);
wmatrix *getDevWekuaActi(wacti *a, wmatrix *b, uint32_t nw, cl_event *be);

void wekuaFreeActi(wacti *a, uint32_t nw, cl_event *be);

// Wekua Network Module
typedef struct {
	void **data; // Module data
	wmatrix **cache; // Cache
	wacti *acti_func; // Activation function
	wmatrix *(*func)(void *, wmatrix *, uint32_t, cl_event *); // Module function
	// void (*get_data)(void *module); // To get module data
	void (*set_cache_id)(void *, int64_t, void *, void *, uint32_t *, int64_t *, wacti **); // To set position in cache
	void (*free_func)(void *, uint32_t, cl_event *); // Free function
	uint8_t com;
	uint32_t nmod, *pseq;
	int64_t arch_id, *w_id; // Position of the output into architecture cache
} wmodule;


// Wekua Network Architecture
typedef struct {
	wmodule **modules; // Modules
	uint32_t nmodule[3]; // Modules number
	wmatrix **weight;
	wmatrix **cache, **s; // Cache
	wacti **acti_funcs;
	wmatrix *(*func)(wmodule **, uint32_t, wmatrix *); // Arch function
	uint32_t pseq;
	int64_t *w_id;
	uint8_t com;
} warch;

warch *wekuaArch(wekuaContext *ctx, uint32_t nmodule, wmatrix *(*func)(wmodule **, uint32_t, wmatrix *), uint8_t com);
void addModuleToArch(warch *arch, wmodule *module);
void configureWekuaArch(warch *arch);
wmatrix *runWekuaArch(warch *arch, wmatrix *input, uint32_t nw, cl_event *be);
void wekuaFreeArch(warch *arch, uint32_t nw, cl_event *be);

// Modules
wmodule *wekuaLinear(wekuaContext *ctx, uint64_t input, uint64_t output, uint32_t deep, wacti *acti, uint8_t com);
wmatrix *runWekuaLinear(void *m, wmatrix *input, uint32_t nw, cl_event *be);
void freeWekuaLinear(void *m, uint32_t nw, cl_event *be);

wmodule *wekuaSequential(wekuaContext *ctx, uint32_t nmodule, uint8_t com);
void addModuleToSequential(wmodule *sequential, wmodule *module);
wmatrix *runWekuaSequential(void *m, wmatrix *input, uint32_t nw, cl_event *be);
void freeWekuaSequential(void *m, uint32_t nw, cl_event *be);

// Optimization functions
typedef struct {
	void **data; // Optim info
	warch *arch;
	void (*step)(void **, warch *, wmatrix *, wmatrix *, wloss *, uint32_t, cl_event *);
} woptim;

woptim *wekuaGradientDescent(double lr, double lri, warch *a);
void wekuaFreeOptimGD(woptim *opti, uint32_t nw, cl_event *be);

woptim *wekuaGradientDescentMomentum(double lr, double lri, double momentum, double imomentum, warch *a);
void wekuaFreeOptimGDM(woptim *opti, uint32_t nw, cl_event *be);

woptim *wekuaAdaGrad(double lr, double lri, warch *a);
void wekuaFreeOptimAdaGrad(woptim *optim, uint32_t nw, cl_event *be);

woptim *wekuaRMSprop(double lr, double lri, double beta, double ibeta, warch *a);
void wekuaFreeOptimRMSprop(woptim *optim, uint32_t nw, cl_event *be);

woptim *wekuaAdaDelta(double lr, double lri, warch *a);
void wekuaFreeOptimAdaDelta(woptim *optim, uint32_t nw, cl_event *be);

void runWekuaOptim(woptim *optim, wmatrix *output, wmatrix *ow, wloss *l, uint32_t nw, cl_event *be);


#endif
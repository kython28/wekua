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

#ifdef __cplusplus
extern "C" {
#endif

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
	cl_device_local_mem_type local_mem_type;
	uint32_t compute_units, clock_frequency, max_work_item_dimensions, vector_width[10];
	uint64_t max_work_group_size, nlen, max_global_size;
} wDevice;

uint32_t getPlatforms(wPlatform **platform);
uint32_t getDevices(wPlatform platform , wDevice **device, cl_device_type type);

void wekuaPlatformFromclPlatform(cl_platform_id platform, wPlatform *plat);
void wekuaDeviceFromclDevice(cl_device_id dev, wDevice *wdev);

void freeWekuaPlatform(wPlatform *plat, uint32_t nplat);
void freeWekuaDevice(wDevice *dev, uint32_t ndev);

typedef struct _wk_ctx {
	cl_context ctx; // OpenCL Context
	cl_command_queue command_queue; // OpenCL Command Queue
	cl_program *programs; // OpenCL programs
	cl_kernel *kernels; // OpenCL kernels
	
	cl_device_id dev; // OpenCL device
	cl_device_local_mem_type local_mem_type;

	// Info
	const uint32_t *dtype_length;
	uint32_t vector_width[10], compute_units;
	uint64_t max_work_group_size;
} *wekuaContext;

wekuaContext createWekuaContext(wDevice *dev, uint8_t use_vectors);
wekuaContext createSomeWekuaContext(cl_device_type type, uint8_t use_vectors);

// Kernels

#define WEKUA_KERNEL_RANDN 0
#define WEKUA_KERNEL_RANDUNIFORM 1
#define WEKUA_KERNEL_IDEN 2
#define WEKUA_KERNEL_TRANS 3
#define WEKUA_KERNEL_AXPY 4
#define WEKUA_KERNEL_SCAL 5
#define WEKUA_KERNEL_DOT 6
#define WEKUA_KERNEL_CONVERT 7
#define WEKUA_KERNEL_ABS 8
#define WEKUA_KERNEL_DIAG 9
#define WEKUA_KERNEL_ARANGE 10
#define WEKUA_KERNEL_POWER 11
#define WEKUA_KERNEL_DIVIDE 12
#define WEKUA_KERNEL_LOG 13
#define WEKUA_KERNEL_SIN 14
#define WEKUA_KERNEL_SINH 15
#define WEKUA_KERNEL_COS 16
#define WEKUA_KERNEL_COSH 17
#define WEKUA_KERNEL_TAN 18
#define WEKUA_KERNEL_TANH 19
#define WEKUA_KERNEL_MUL 20
#define WEKUA_KERNEL_FILL 21
#define WEKUA_KERNEL_EULER_IDEN 22
#define WEKUA_KERNEL_ROOT_DEV 23
#define WEKUA_KERNEL_ROOT 24
#define WEKUA_KERNEL_DET 25
#define WEKUA_KERNEL_GAUSS 26
#define WEKUA_KERNEL_GAUSS_2 27
#define WEKUA_KERNEL_BIAS 28
#define WEKUA_KERNEL_SIGMOID 29
#define WEKUA_KERNEL_GEMM 30
#define WEKUA_KERNEL_SUM 31
#define WEKUA_KERNEL_LINEAR_BIAS_STEP 32
#define WEKUA_KERNEL_SQRT 33
#define WEKUA_KERNEL_ADAGRAD 34
#define WEKUA_KERNEL_GDM 35
#define WEKUA_KERNEL_RMSPROP 36
#define WEKUA_KERNEL_ADADELTA 37
#define WEKUA_KERNEL_RELU 38
#define WEKUA_KERNEL_RELU_DEV 39
#define WEKUA_KERNEL_LEAKY_RELU 40
#define WEKUA_KERNEL_LEAKY_RELU_DEV 41
#define WEKUA_KERNEL_MSE 42
#define WEKUA_KERNEL_SIGMOID_DEV 43
#define WEKUA_KERNEL_TANH_DEV 44
#define WEKUA_KERNEL_ADAM 45

cl_kernel compileKernel(wekuaContext ctx, uint8_t id, uint8_t dtype, uint8_t com);

void freeWekuaContext(wekuaContext context);

// To get random buffer from /dev/urandom
void getRandomBuffer(void *buf, uint64_t size);

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

typedef struct _wk_matrix {
	cl_mem real; // Real numbers
	cl_mem imag; // Imaginary numbers

	void *raw_real; // Real numbers array mapped
	void *raw_imag; // Imaginary numbers array mapped

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
uint8_t createComplexMatrix(wmatrix a); // To enable complex numbers.
int removeComplexMatrix(wmatrix b, uint32_t nw, cl_event *be); // To disable complex numbers.

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

void wekuaGetValueFromMatrix(wmatrix a, uint64_t y, uint64_t x, void *real, void *imag, uint32_t nw, cl_event *be);
void wekuaPutValueToMatrix(wmatrix a, uint64_t y, uint64_t x, void *real, void *imag, uint32_t nw, cl_event *be);

int wekuaCopyMatrixRegion(
	wmatrix src, uint64_t src_offset_x, uint64_t src_offset_y,
	wmatrix dst, uint64_t dst_offset_x, uint64_t dst_offset_y,
	uint64_t w, uint64_t h
);

// Some BLAS functions
int wekuaBlasAxpy(wmatrix x, wmatrix y, void *alpha, void *beta, uint32_t nw, cl_event *be, cl_event *e); // y = (alpha+beta*j)*x + y
int wekuaBlasScalar(wmatrix x, void *alpha, void *beta, uint32_t nw, cl_event *be, cl_event *e); // Dot all elements in a matrix for a scalar. Alpha is real number and Beta is imaginary number
int wekuaBlasGemm(void *ralpha, void *ialpha, uint8_t a_trans, wmatrix a, uint8_t b_trans, wmatrix b,
	void *rbeta, void *ibeta, wmatrix c, uint32_t nw, cl_event *be
);
 
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

int wekuaMatrixAdd(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e); // Matrix addition
int wekuaMatrixSub(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e); // Matrix Substration
int wekuaMatrixDot(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e); // Hadamard product
int wekuaMatrixDivide(wmatrix a, wmatrix b, uint32_t nw, cl_event *be, cl_event *e); // a_{i} /= b_{i}
int wekuaMatrixPower(wmatrix a, wmatrix b, void *exp_r, void *exp_i, uint32_t nw, cl_event *be, cl_event *e); // a_{i} = a_{i}^{exp_r+exp_i*i} or a_{i} = a_{i}^{b_{i}}
int wekuaMatrixLn(wmatrix a, uint32_t nw, cl_event *be, cl_event *e); // a_{i} = ln(a_{i})
int wekuaMatrixLog(wmatrix a, wmatrix b, void *base_r, void *base_i);
int wekuaMatrixTrace(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be); // Matrix Trace
int wekuaMatrixSqrt(wmatrix a, uint32_t nw, cl_event *be, cl_event *e); // a_{i} = sqrt(a_{i})

// Linear functions
wmatrix wekuaMatrixInv(wmatrix a, uint32_t nw, cl_event *be); // Matrix inverse
wmatrix wekuaMatrixSolve(wmatrix a, wmatrix b, uint32_t nw, cl_event *be);
wmatrix wekuaMatrixPinv(wmatrix a, uint32_t nw, cl_event *be); // Matrix Pseudo-inverse
uint64_t wekuaMatrixRang(wmatrix a, uint32_t nw, cl_event *be); // Matrix range
int wekuaMatrixDet(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be); // Matrix determinant

// Trigonometric functions
int wekuaMatrixSin(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);
int wekuaMatrixCos(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);
int wekuaMatrixTan(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);
int wekuaMatrixSinh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);
int wekuaMatrixCosh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);
int wekuaMatrixTanh(wmatrix a, uint32_t nw, cl_event *be, cl_event *e);

// Extra functions
int wekuaMatrixSum(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be); // Sum of all the elements
int wekuaMatrixMul(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be); // Mul of all the elements
int wekuaMatrixMean(wmatrix a, void *real, void *imag, uint32_t nw, cl_event *be); // Mean of all the elements
void wekuaMatrixMax(wmatrix a, uint64_t *y, uint64_t *x, uint32_t nw, cl_event *be); // To get max value.
void wekuaMatrixMin(wmatrix a, uint64_t *y, uint64_t *x, uint32_t nw, cl_event *be); // To get min value.
wmatrix wekuaMatrixPoly(wmatrix a, uint32_t nw, cl_event *be); // Matrix Poly (Leverrier)
wmatrix wekuaMatrixEulerIden(wmatrix angle, uint32_t nw, cl_event *be);
wmatrix wekuaMatrixRoot(wmatrix a, uint32_t nw, cl_event *be); // Polynomial roots

int wekuaFreeMatrix(wmatrix a, uint32_t nw, cl_event *be); // To free a matrix

uint8_t saveWekuaMatrix(const char *name, wmatrix a);
wmatrix loadWekuaMatrix(const char *name, wekuaContext ctx);

// Deep Learning section

typedef struct _w_cache {
	uint64_t ndata;
	void *data;
} *wcache;

typedef struct _w_error {
	wmatrix err; // Error derivate
	void *o_err; // Other errors :v
	uint64_t no_err;
} *werror;

int wekuaMSE(wmatrix output, wmatrix output_wanted, void *error, void *errori, werror *err, uint32_t nw, cl_event *be); // Mean Square Error

typedef struct _w_acti {
	void *data; // Activation function data
	int (*run_acti)(void *, wmatrix, uint32_t, cl_event *); // To run the activation function
	wmatrix (*get_dev)(void *, wmatrix); // To get derivate
	void (*free_func)(void *, uint32_t, cl_event *); // To free the wacti object
} *wacti;

wacti wekuaActiLinear(); // -> x
wacti wekuaActiTanh(); // -> wekuaMatrixTanh(x)
wacti wekuaActiSigmoid(); // -> 1/(1 + e^(-x))
wacti wekuaActiReLU(); // -> max(0, x)
wacti wekuaActiLeakyReLU(wekuaContext ctx, void *alpha, void *alphai, uint8_t dtype); // -> max(alpha*x, x) & (0.0 < alpha < 1.0)
// wacti wekuaActiELU(); // -> alpha*(e^x - 1)

int runWekuaActi(wacti acti, wmatrix input, uint32_t nw, cl_event *be);
wmatrix wekuaActiGetDev(wacti acti, wmatrix output);
void wekuaFreeActi(wacti acti, uint32_t nw, cl_event *be);

typedef struct _w_neuron {
	wmatrix *weight; // Neuron weight
	wmatrix *bias; // Neuron bias
	uint64_t layer; // Layer num
	uint8_t dtype; // Weight data type
	wacti acti; // Activation function for the neuron

	// To run the neuron
	void* (*run)(void *, void *, wcache *, uint32_t, cl_event *);
	int (*backward)(void *, werror error, wcache cache, werror *err);
	int (*step)(void *, void *, void *, werror error, wcache cache, int (*)(void *, void *, uint32_t, wmatrix, wmatrix, wmatrix, wmatrix));

	void (*free_error)(werror);
	void (*free_cache)(wcache);
} *wneuron;

wneuron wekuaLinear(wekuaContext ctx, uint64_t input, uint64_t output, uint64_t deep, uint8_t bias, wacti acti, uint8_t dtype);


// typedef struct _w_conv_inputs {
// 	uint32_t num; // Samples number
// 	wmatrix *sample;
// } wconvinput;

// wneuron wekuaConv1d(wekuaContext ctx, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint8_t bias, wacti acti, uint8_t dtype);
// wneuron wekuaConv2d(wekuaContext ctx, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size_w, uint64_t kernel_size_h, uint64_t stride, uint8_t bias);

void *runWekuaNeuron(wneuron neuron, void *input, wcache *cache, uint32_t nw, cl_event *be);

int wekuaNeuronBackward(wneuron neuron, werror error, wcache cache, werror *err);

uint8_t saveWekuaNeuron(const char *name, wneuron neuron);
uint8_t loadWekuaNeuron(const char *name, wneuron neuron);

void wekuaFreeNeuron(wneuron neur, uint32_t nw, cl_event *be);

typedef struct _w_net {
	wneuron *neurons;
	uint32_t nneur;
	uint8_t dtype;
} *wnetwork;

wnetwork wekuaNeuronNetwork(uint32_t neur_num, uint8_t dtype);
void *runWekuaNetwork(wnetwork net, void *input, wcache **cache);
int wekuaNetworkBackward(wnetwork net, werror *error, wcache *cache, werror *err);

uint8_t saveWekuaNetwork(const char *name, wnetwork net);
uint8_t loadWekuaNetwork(const char *name, wnetwork net, wekuaContext ctx);

void wekuaFreeNetCache(wnetwork net, wcache *cache);
void wekuaFreeNetError(wnetwork net, werror *error);
void wekuaFreeNetwork(wnetwork net, uint32_t nw, cl_event *be);

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
woptim wekuaOptimAdam(wekuaContext ctx,  wnetwork net, void *lr, void *lri, void *beta1, void *beta1i, void *beta2, void *beta2i, uint8_t dtype);

int wekuaOptimStep(woptim optim, werror *error, wcache *cache);
int wekuaOptimZero(woptim optim);

void wekuaFreeOptim(woptim optim, uint32_t nw, cl_event *be);

#ifdef __cplusplus
}
#endif

#endif
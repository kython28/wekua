#ifndef WEKUA_H
#define WEKUA_H

#define CL_TARGET_OPENCL_VERSION 300

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdint.h>
#include "fifo.h"



#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	cl_platform_id platform; // Platform ID
	uint8_t *name; // Platform name
	uint64_t nlen; // Devices numbers
} wPlatform;

typedef struct {
	wPlatform *platform; // Platform

	cl_device_id device; // Device ID
	cl_device_type type; // Device type
	uint8_t *name; // Device name
	// Device Info
	cl_device_local_mem_type local_mem_type;
	uint32_t compute_units, clock_frequency, max_work_item_dimensions, vector_width[10];
	uint64_t max_work_group_size, nlen, max_global_size;
} wDevice;

uint32_t getPlatforms(wPlatform **platform);
uint32_t getDevices(wPlatform *platform , wDevice **device, cl_device_type type);

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
	cl_mem_flags flags;

	// Garbage collector
	pthread_t garbage_collector;
	void *garbage_collector_arg;
	wfifo garbage_queue;

	// Info
	const uint32_t *dtype_length;
	uint32_t vector_width[10], compute_units;
	uint64_t max_work_group_size;
} *wekuaContext;

wekuaContext createWekuaContext(wDevice *dev, uint8_t use_vectors, uint8_t alloc_host_mem);
wekuaContext createSomeWekuaContext(cl_device_type type, uint8_t use_vectors, uint8_t alloc_host_mem);
wekuaContext createWekuaContextsFromOpenCLContext(cl_context ctx, wDevice *dev, uint8_t use_vectors, uint8_t alloc_host_mem);

// Kernel list
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
#define WEKUA_KERNEL_SCALAR_ADD 46
#define WEKUA_KERNEL_L1_REGULARIZATION 47
#define WEKUA_KERNEL_REGULARIZATION 48

cl_kernel compileKernel(wekuaContext ctx, uint8_t id, uint8_t dtype, uint8_t com);
cl_kernel compileCustomKernel(wekuaContext ctx, const char *filename, const char *kernel_name, char *args, cl_program *program);

void freeWekuaContext(wekuaContext context);

#ifdef __cplusplus
}
#endif
#endif

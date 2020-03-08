#ifndef WEKUA_H
#define WEKUA_H

#include <CL/cl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define WEKUA_DEVICE_TYPE_CPU CL_DEVICE_TYPE_CPU
#define WEKUA_DEVICE_TYPE_GPU CL_DEVICE_TYPE_GPU
#define WEKUA_DEVICE_TYPE_ALL CL_DEVICE_TYPE_ALL
#define WEKUA_DEVICE_TYPE_ACCELERATOR CL_DEVICE_TYPE_ACCELERATOR
#define WEKUA_DEVICE_TYPE_CUSTOM CL_DEVICE_TYPE_CUSTOM
#define WEKUA_DEVICE_TYPE_DEFAULT CL_DEVICE_TYPE_DEFAULT

typedef cl_device_type wekua_device_type;

typedef struct {
	cl_platform_id platform;
	uint8_t *name;
	uint64_t nlen;
} wPlatform;

typedef struct {
	cl_device_id device;
	cl_device_type type;
	uint8_t *name;
	uint32_t compute_units, clock_frequency, max_work_item_dimensions;
	uint64_t max_work_group_size, *max_work_item_sizes, nlen, max_size;
} wDevice;

uint32_t getPlatforms(wPlatform **platform);
uint32_t getDevices(wPlatform platform , wDevice **device, wekua_device_type type);
void freeWekuaPlatform(wPlatform *plat);
void freeWekuaDevice(wDevice *dev, uint32_t ndev);

typedef struct {
	cl_context ctx;
	cl_command_queue command_queue;
	cl_program *programs;
	cl_kernel *kernels;
	uint64_t max_work_item_dimensions;
	uint64_t max_work_group_size, *max_work_item_sizes;
} wekuaContext;

wekuaContext *createWekuaContext(wDevice *dev);
void freeWekuaContext(wekuaContext *context);


// Tensor

typedef struct {
	cl_mem data;
	double *raw_data;
	uint32_t *shape, dim;
	uint64_t size;
} wTensor;

wTensor *wekuaAllocTensor(wekuaContext *ctx, uint32_t dim, uint32_t *shape, double alpha); // x = alpha*e
void wekuaFreeTensor(wekuaContext *ctx, wTensor *tensor);

wTensor *wekuaTensorCopy(wekuaContext *ctx, wTensor *a);  // y = 0*y + x
void wekuaTensorAdd(wekuaContext *ctx, wTensor *a, wTensor *b); // y = 1*x + y
void wekuaTensorSub(wekuaContext *ctx, wTensor *a, wTensor *b); // y = -1*x + y
void wekuaTensorDot(wekuaContext *ctx, wTensor *a, double alpha); // x = alpha*x

#endif
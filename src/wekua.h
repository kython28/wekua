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
void freeWekuaPlatform(wPlatform *plat, uint32_t nplat);
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

void getRandomBuffer(void *buf, uint64_t size);

// Matrix

typedef struct {
	cl_mem data;
	double *raw_data;
	wekuaContext *ctx;
	uint32_t r,c;
	uint64_t size, work_items[3];
} wMatrix;

wMatrix *wekuaAllocMatrix(wekuaContext *ctx, uint32_t r, uint32_t c, double alpha);
wMatrix *wekuaAllocMatrixRand(wekuaContext *ctx, uint32_t r, uint32_t c);
wMatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint32_t r, uint32_t c, void *buf);
void wekuaFreeMatrix(wMatrix *Matrix);

double wekuaMatrixGet(wMatrix *a, uint32_t x, uint32_t y);
void wekuaMatrixPut(wMatrix *a, uint32_t x, uint32_t y, double n);

wMatrix *wekuaMatrixCopy(wMatrix *a);  // y = 0*y + x
void wekuaMatrixAdd(wMatrix *a, wMatrix *b); // y = 1*x + y
void wekuaMatrixSub(wMatrix *a, wMatrix *b); // y = -1*x + y
void wekuaMatrixDot(wMatrix *a, double alpha); // x = alpha*x
wMatrix *wekuaMatrixTrans(wMatrix *a);
wMatrix *wekuaMatrixIden(wekuaContext *ctx, uint32_t c);
wMatrix *wekuaSubMatrix(wMatrix *a, uint32_t x, uint32_t w, uint32_t y, uint32_t h);
wMatrix *wekuaMatrixProduct(wMatrix *a, wMatrix *b);

#endif

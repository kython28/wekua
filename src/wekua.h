#ifndef WEKUA_H
#define WEKUA_H

#include <CL/cl.h>
#include <stdint.h>

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
	uint64_t max_work_group_size, *max_work_item_sizes, nlen;
} wDevice;

uint32_t getPlatforms(wPlatform **platform);
uint32_t getDevices(wPlatform platform , wDevice **device, wekua_device_type type);
void freeWekuaPlatform(wPlatform *plat);
void freeWekuaDevice(wDevice *dev);

typedef struct {
	cl_context ctx;
	cl_command_queue command_queue;
	cl_program *programs;
	cl_kernel *kernels;
} wekuaContext;

wekuaContext *createWekuaContext(wDevice *dev);




#endif
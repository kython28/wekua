#ifndef WEKUA_H
#define WEKUA_H

#define CL_TARGET_OPENCL_VERSION 300

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "utils/fifo.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _w_platform_t {
	cl_platform_id platform; // OpenCL platform ID
	char *name; // Platform name
	size_t len; // Name length
} *wplatform;

typedef struct _w_device_t {
	cl_device_id device; // OpenCL device ID
	cl_device_type device_type; // OpenCL device type
	char *name; // Device name
	uint64_t len; // Name length

	// Device information
	cl_device_local_mem_type local_mem_type;
	uint32_t compute_units;
	uint32_t clock_frequency;
	uint32_t max_work_item_dimensions;
	uint32_t vectors_size[10];
	uint64_t max_work_group_size;
	uint64_t max_global_size;
} *wdevice;

uint32_t wekuaGetOpenCLPlatforms(wplatform **platforms);
uint32_t wekuaGetOpenCLDevices(wplatform platform, wdevice **devices, cl_device_type type);

wplatform wekuaPlatformFromclPlatform(cl_platform_id platform);
wdevice wekuaDeviceFromclDevice(cl_device_id dev);

void freeWekuaPlatforms(wplatform *platforms, uint32_t nplat);
void freeWekuaDevices(wdevice *devices, uint32_t ndev);

typedef struct _w_kernel {
	void (*release_cl_kernels)(struct _w_kernel*);
} *wkernel;

typedef struct _wk_ctx {
	cl_context ctx; // OpenCL Context
	cl_command_queue command_queue; // OpenCL Command Queue
	wkernel *kernels; // Wekua Kernels

	struct _w_device_t device; // Wekua device

	// Some informations
	const uint32_t *dtype_length;
	cl_mem_flags mem_flags;

	// Process
	pthread_t memory_allocator[2];
	pthread_t memory_destroyer[2];

	wfifo memory_allocator_queue;
	wfifo memory_destroyer_queue;
} *wekuaContext;

wekuaContext createWekuaContext(wdevice device, uint8_t use_vectors, uint8_t alloc_host_mem);
wekuaContext createSomeWekuaContext(cl_device_type type, uint8_t use_vectors, uint8_t alloc_host_mem);
void freeWekuaContext(wekuaContext context);

// Kernels list

cl_kernel compileKernel(wekuaContext ctx, uint8_t id, uint8_t com);
cl_kernel compileCustomKernel(wekuaContext ctx, const char *filename, const char *kernel_name, char *args, cl_program *program);

#ifdef __cplusplus
}
#endif
#endif
#include "../headers/wekua.h"

#include <stdio.h>

#include <string.h>

#define WEKUA_OPENCL_GET_ATTR_SIZE(func, obj, attr, ptr, error_label) \
	ret = func(obj, attr, 0, NULL, ptr); \
	if (ret != CL_SUCCESS) goto error_label; \

#define WEKUA_OPENCL_GET_PLATFORM_ATTR_SIZE(plat, attr, ptr, error_label) WEKUA_OPENCL_GET_ATTR_SIZE(clGetPlatformInfo, plat, attr, ptr, error_label)
#define WEKUA_OPENCL_GET_DEVICE_ATTR_SIZE(dev, attr, ptr, error_label) WEKUA_OPENCL_GET_ATTR_SIZE(clGetDeviceInfo, dev, attr, ptr, error_label)

#define WEKUA_OPENCL_GET_INFO(func, obj, attr, size, ptr, error_label) \
	ret = func(obj, attr, size, ptr, NULL); \
	if (ret != CL_SUCCESS) goto error_label; \

#define WEKUA_OPENCL_GET_PLATFORM_INFO(plat, attr, size, ptr, error_label) WEKUA_OPENCL_GET_INFO(clGetPlatformInfo, plat, attr, size, ptr, error_label)
#define WEKUA_OPENCL_GET_DEVICE_INFO(dev, attr, size, ptr, error_label) WEKUA_OPENCL_GET_INFO(clGetDeviceInfo, dev, attr, size, ptr, error_label)

uint32_t wekuaGetOpenCLPlatforms(wplatform **platforms){
	if (!platforms) return 0;

	uint32_t nplat = 0;
	cl_int ret;

	*platforms = NULL;
	cl_platform_id *plat = NULL;

	ret = clGetPlatformIDs(0, NULL, &nplat);
	if (ret == CL_SUCCESS || nplat > 0){
		plat = (cl_platform_id*) calloc(nplat, sizeof(cl_platform_id));
		if (!plat) goto wekuaGetOpenCLPlatforms_error;

		*platforms = (wplatform*) calloc(nplat, sizeof(wplatform));
		if (!(*platforms)) goto wekuaGetOpenCLPlatforms_error;

		ret = clGetPlatformIDs(nplat, plat, NULL);
		if (ret != CL_SUCCESS) goto wekuaGetOpenCLPlatforms_error;

		for (uint32_t x=0; x<nplat; x++){
			wplatform p = wekuaPlatformFromclPlatform(plat[x]);
			if(!p) goto wekuaGetOpenCLPlatforms_error;

			(*platforms)[x] = p;
		}
		free(plat);
	}
	return nplat;

	wekuaGetOpenCLPlatforms_error:
	if (plat) free(plat);
	if (*platforms) freeWekuaPlatforms(*platforms, nplat);

	return 0;
}

uint32_t wekuaGetOpenCLDevices(wplatform platform, wdevice **devices, cl_device_type type){
	if (!platform || !devices) return 0;

	uint32_t ndev = 0;

	cl_platform_id platform_id = platform->platform;
	cl_device_id *dev = NULL;
	*devices = NULL;

	cl_int ret = clGetDeviceIDs(platform_id, type, 0, NULL, &ndev);
	if (ret == CL_SUCCESS || ndev > 0){
		dev = (cl_device_id*) calloc(ndev, sizeof(cl_device_id));
		if (!dev) goto wekuaGetOpenCLDevices_error;

		*devices = (wdevice*) calloc(ndev, sizeof(wdevice));
		if (!(*devices)) goto wekuaGetOpenCLDevices_error;

		ret = clGetDeviceIDs(platform_id, type, ndev, dev, NULL);
		if (ret != CL_SUCCESS) goto wekuaGetOpenCLDevices_error;

		for (uint32_t x=0; x<ndev; x++){
			wdevice d = wekuaDeviceFromclDevice(dev[x]);
			if (!d) goto wekuaGetOpenCLDevices_error;

			dev[x] = NULL;
			(*devices)[x] = d;
		}
		free(dev);
	}
	return ndev;

	wekuaGetOpenCLDevices_error:
	if (dev) {
		for (uint32_t x=0; x<ndev; x++) clReleaseDevice(dev[x]);
		free(dev);
	}
	if (*devices) freeWekuaDevices(*devices, ndev);

	return 0;
}

wplatform wekuaPlatformFromclPlatform(cl_platform_id platform){
	if (!platform) return NULL;

	wplatform _platform = (wplatform) calloc(1, sizeof(struct _w_platform_t));
	if (!_platform) return NULL;

	_platform->platform = platform;
	size_t len = 0; char *name = NULL;
	cl_int ret;

	WEKUA_OPENCL_GET_PLATFORM_ATTR_SIZE(platform, CL_PLATFORM_NAME, &len, wekuaPlatformFromclPlatform_error)
	else if (len == 0) goto wekuaPlatformFromclPlatform_error;

	name = (char*) calloc(1, len+1);
	if (!name) goto wekuaPlatformFromclPlatform_error;
	_platform->name = name;

	WEKUA_OPENCL_GET_PLATFORM_INFO(platform, CL_PLATFORM_NAME, len, name, wekuaPlatformFromclPlatform_error)
	return _platform;

	wekuaPlatformFromclPlatform_error:
	if (name) free(name);
	free(_platform);
	return NULL;
}

wdevice wekuaDeviceFromclDevice(cl_device_id dev){
	if (!dev) return NULL;

	wdevice wdev = (wdevice) calloc(1, sizeof(struct _w_device_t));
	if (!wdev) return NULL;

	wdev->device = dev;
	cl_int ret;

	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &wdev->device_type, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, sizeof(uint32_t), &wdev->compute_units, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uint32_t), &wdev->clock_frequency, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(uint32_t), &wdev->max_work_item_dimensions, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(uint64_t), &wdev->max_work_group_size, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(uint64_t), &wdev->max_global_size, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &wdev->local_mem_type, wekuaDeviceFromclDevice_error)
	// WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(uint64_t), &wdev->max_work_item_dimensions, wekuaDeviceFromclDevice_error)

	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(uint32_t), wdev->vectors_size, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(uint32_t), wdev->vectors_size + 1, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(uint32_t), wdev->vectors_size + 2, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(uint32_t), wdev->vectors_size + 3, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(uint32_t), wdev->vectors_size + 4, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(uint32_t), wdev->vectors_size + 5, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(uint32_t), wdev->vectors_size + 6, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(uint32_t), wdev->vectors_size + 7, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(uint32_t), wdev->vectors_size + 8, wekuaDeviceFromclDevice_error)
	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(uint32_t), wdev->vectors_size + 9, wekuaDeviceFromclDevice_error)

	uint64_t len = 0;
	WEKUA_OPENCL_GET_DEVICE_ATTR_SIZE(dev, CL_DEVICE_NAME, &len, wekuaDeviceFromclDevice_error)
	else if (len == 0) goto wekuaDeviceFromclDevice_error;

	char *name = (char*) calloc(1, len+1);
	if (!name) goto wekuaDeviceFromclDevice_error;

	wdev->name = name;
	wdev->len = len;

	WEKUA_OPENCL_GET_DEVICE_INFO(dev, CL_DEVICE_NAME, len, name, wekuaDeviceFromclDevice_error)

	return wdev;

	wekuaDeviceFromclDevice_error:
	if (wdev->name) free(wdev->name);
	free(wdev);

	return NULL;
}

void freeWekuaPlatforms(wplatform *platforms, uint32_t nplat){
	if (!platforms) return;

	for (uint32_t x=0; x<nplat; x++){
		wplatform platform = platforms[x];

		free(platform->name);
		free(platform);
	}
	free(platforms);
}

static void freeWekuaDevice(wdevice device){
	if (device->name) free(device->name);
	if (device->device) clReleaseDevice(device->device);
}

void freeWekuaDevices(wdevice *devices, uint32_t ndev){
	if (!devices) return;

	for (uint32_t x=0; x<ndev; x++){
		wdevice device = devices[x];
		freeWekuaDevice(device);
		free(device);
	}
	free(devices);
}

const uint32_t dtype_length[10] = {
	sizeof(int8_t), sizeof(uint8_t), // int8_t
	sizeof(int16_t), sizeof(uint16_t), // int16_t
	sizeof(int32_t), sizeof(uint32_t), // int32_t
	sizeof(int64_t), sizeof(uint64_t), // int64_t
	sizeof(float),
	sizeof(double)
};

#define WEKUA_KERNEL_NUM 1
#define KERNEL_COL 10*WEKUA_KERNEL_NUM

wekuaContext createWekuaContext(wdevice dev, uint8_t use_vectors, uint8_t alloc_host_mem){
	if (!dev) return NULL;
	else if (dev->max_work_item_dimensions < 3) return NULL;

	wekuaContext context = (wekuaContext) calloc(1, sizeof(struct _wk_ctx));
	if (!context) return NULL;

	wdevice device = &context->device;
	memcpy(device, dev, sizeof(struct _w_device_t));
	memset(dev, 0, sizeof(struct _w_device_t));

	cl_int ret;
	context->ctx = clCreateContext(NULL, 1, &device->device, NULL, NULL, &ret);
	if (ret != CL_SUCCESS) goto createWekuaContext_error;

	context->command_queue = clCreateCommandQueueWithProperties(context->ctx, device->device, NULL, &ret);
	if (ret != CL_SUCCESS) goto createWekuaContext_error;

	context->kernels = (wkernel*) calloc(KERNEL_COL, sizeof(void*));
	context->dtype_length = dtype_length;

	if (alloc_host_mem) context->mem_flags = CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR;
	else context->mem_flags = CL_MEM_READ_WRITE;

	if (!use_vectors) {
		for (uint8_t x=0; x<10; x++) device->vectors_size[x] = 1;
	}

	return context;

	createWekuaContext_error:
	freeWekuaContext(context);
	return NULL;
}

wekuaContext createSomeWekuaContext(cl_device_type type, uint8_t use_vectors, uint8_t alloc_host_mem){
	wplatform *platforms = NULL;
	uint32_t nplat = wekuaGetOpenCLPlatforms(&platforms);
	if (nplat == 0) return NULL;

	wdevice *devices = NULL; uint32_t ndev = 0;
	uint64_t best_device_points = 0;

	struct _w_device_t best_device;
	for (uint32_t x=0; x<nplat; x++){
		ndev = wekuaGetOpenCLDevices(platforms[x], &devices, type);
		if (ndev == 0) continue;

		for (uint32_t y=0; y<ndev; y++){
			wdevice device = devices[y];
			uint64_t points = device->clock_frequency*device->compute_units*device->max_work_group_size;

			if (points > best_device_points){
				freeWekuaDevice(&best_device);

				memcpy(&best_device, device, sizeof(struct _w_device_t));
				memset(device, 0, sizeof(struct _w_device_t));
			}
		}
		freeWekuaDevices(devices, ndev);
	}

	wekuaContext ctx = createWekuaContext(&best_device, use_vectors, alloc_host_mem);

	freeWekuaDevice(&best_device);
	freeWekuaPlatforms(platforms, nplat);

	return ctx;
}

void freeWekuaContext(wekuaContext context){
	if (!context) return;

	if (context->kernels){
		wkernel *kernels = context->kernels;
		for (uint32_t x=0; x<KERNEL_COL; x++){
			wkernel k = kernels[x];
			if (k) k->release_cl_kernels(k);
		}
		free(context->kernels);
	}

	if (context->command_queue){
		clFinish(context->command_queue);
		clReleaseCommandQueue(context->command_queue);
	}

	if (context->device.device) freeWekuaDevice(&context->device);
	if (context->ctx) clReleaseContext(context->ctx);
	free(context);
}

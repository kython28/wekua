#include "wekua.h"
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define KERNEL_NUM 0
#define KEID 8

uint8_t kernels[][20] = {
	"kernels/"
};

uint32_t getPlatforms(wPlatform **platform){
	uint32_t nplat;
	clGetPlatformIDs(0, NULL, &nplat);
	if (nplat > 0){
		cl_platform_id *plat = (cl_platform_id*) calloc(nplat, sizeof(cl_platform_id));
		*platform = (wPlatform*) calloc(nplat, sizeof(wPlatform));
		clGetPlatformIDs(nplat, plat, NULL);
		for (uint32_t x=0; x<nplat; x++){
			(*platform)[x].platform = plat[x];
			clGetPlatformInfo(plat[x], CL_PLATFORM_NAME, 0, NULL, &(*platform)[x].nlen);
			(*platform)[x].name = (uint8_t*) calloc((*platform)[x].nlen, 1);
			clGetPlatformInfo(plat[x], CL_PLATFORM_NAME, (*platform)[x].nlen, (*platform)[x].name, NULL);
		}
		free(plat);
	}
	return nplat;
}

void freeWekuaPlatform(wPlatform *plat){
	free(plat->name);
}

uint32_t getDevices(wPlatform platform , wDevice **device, wekua_device_type type){
	uint32_t ndev;
	clGetDeviceIDs(platform.platform, type, 0, NULL, &ndev);
	if (ndev > 0){
		cl_device_id *dev = (cl_device_id*) calloc(ndev, sizeof(cl_device_id));
		*device = (wDevice*) calloc(ndev, sizeof(wDevice));
		clGetDeviceIDs(platform.platform, type, ndev, dev, NULL);
		for (uint32_t x=0; x<ndev; x++){
			(*device)[x].device = dev[x];
			uint64_t s;
			clGetDeviceInfo(dev[x], CL_DEVICE_TYPE, 0, NULL, &s);
			clGetDeviceInfo(dev[x], CL_DEVICE_TYPE, s, &(*device)[x].type, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &s);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_COMPUTE_UNITS, s, &(*device)[x].compute_units, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_CLOCK_FREQUENCY, 0, NULL, &s);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_CLOCK_FREQUENCY, s, &(*device)[x].clock_frequency, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 0, NULL, &s);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, s, &(*device)[x].max_work_item_dimensions, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &s);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_ITEM_SIZES, s, (*device)[x].max_work_item_sizes, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &s);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_GROUP_SIZE, s, &(*device)[x].max_work_group_size, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_NAME, 0, NULL, &(*device)[x].nlen);
			(*device)[x].name = (uint8_t*) calloc((*device)[x].nlen, 1);
			clGetDeviceInfo(dev[x], CL_DEVICE_NAME, s, (*device)[x].name, NULL);
		}
		free(dev);
	}
	return ndev;
}

void freeWekuaDevice(wDevice *dev){
	free(dev->name);
	free(dev->max_work_item_sizes);
}

uint8_t *getKernelData(const uint8_t *name, uint64_t *size){
	int fd = open(name, O_RDONLY);
	if (fd < 0){
		return NULL;
	}
	*size = lseek(fd, 0, SEEK_END);
	lseek(fd, 0, SEEK_SET);
	uint8_t *cont = (uint8_t*) calloc(size[0], 1);
	if (write(fd, cont, size[0]) != size[0]){
		free(cont);
		return NULL;
	}
	return cont;
}

wekuaContext *createWekuaContext(wDevice *dev){
	wekuaContext *context = (wekuaContext*) calloc(1, sizeof(wekuaContext));
	context->ctx = clCreateContext(NULL, 1, &dev->device, NULL, NULL, NULL);
	context->command_queue = clCreateCommandQueueWithProperties(context->ctx, dev->device, NULL, NULL);
	context->programs = (cl_program*) calloc(KERNEL_NUM, sizeof(cl_program));
	context->kernels = (cl_kernel*) calloc(KERNEL_NUM, sizeof(cl_kernel));
	for (uint32_t x=0; x<KERNEL_NUM; x++){
		uint64_t size;
		uint8_t *source = getKernelData(kernels[x], &size);
		context->programs[x] = clCreateProgramWithSource(context->ctx, 1, &source, &size, NULL);
		clBuildProgram(context->programs[x], 1, &dev->device, NULL, NULL, NULL);
		context->kernels[x] = clCreateKernel(context->programs[x], &kernels[x][KEID], NULL);
	}
	return context;
}
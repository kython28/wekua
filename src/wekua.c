#include "wekua.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define KERNEL_NUM 33

const char kernels[KERNEL_NUM][40] = {
	"/usr/lib/wekua_kernels/rand.cl",
	"/usr/lib/wekua_kernels/iden.cl",
	"/usr/lib/wekua_kernels/trans.cl",
	"/usr/lib/wekua_kernels/cut.cl",
	"/usr/lib/wekua_kernels/axpy.cl",
	"/usr/lib/wekua_kernels/product.cl",
	"/usr/lib/wekua_kernels/sin.cl",
	"/usr/lib/wekua_kernels/cos.cl",
	"/usr/lib/wekua_kernels/tan.cl",
	"/usr/lib/wekua_kernels/sinh.cl",
	"/usr/lib/wekua_kernels/cosh.cl",
	"/usr/lib/wekua_kernels/tanh.cl",
	"/usr/lib/wekua_kernels/dotscalar.cl",
	"/usr/lib/wekua_kernels/abs.cl",
	"/usr/lib/wekua_kernels/diag.cl",
	"/usr/lib/wekua_kernels/sum.cl",
	"/usr/lib/wekua_kernels/mul.cl",
	"/usr/lib/wekua_kernels/norm.cl",
	"/usr/lib/wekua_kernels/resize.cl",
	"/usr/lib/wekua_kernels/det.cl",
	"/usr/lib/wekua_kernels/gauss.cl",
	"/usr/lib/wekua_kernels/gauss2.cl",
	"/usr/lib/wekua_kernels/randuniform.cl",
	"/usr/lib/wekua_kernels/aberth.cl",
	"/usr/lib/wekua_kernels/logsig.cl",
	"/usr/lib/wekua_kernels/relu.cl",
	"/usr/lib/wekua_kernels/leakyrelu.cl",
	"/usr/lib/wekua_kernels/softplus.cl",
	"/usr/lib/wekua_kernels/log.cl",
	"/usr/lib/wekua_kernels/dot.cl",
	"/usr/lib/wekua_kernels/divide.cl",
	"/usr/lib/wekua_kernels/power.cl",
	"/usr/lib/wekua_kernels/arange.cl"
};

const char ker_name[KERNEL_NUM][20] = {
	"rand", "iden", "trans", "cut", "axpy",
	"product", "sen", "cose", "tg", "senh", "coseh", "tgh",
	"dots", "absolute", "diag", "sum", "mul", "norm",
	"resize", "det", "gauss", "gauss2", "uniform",
	"aberth", "logsig", "relu", "lerelu", "softplus",
	"lognatu", "dotm", "divide", "power", "arange"
};

void getRandomBuffer(void *buf, uint64_t size){
	int fd = open("/dev/urandom", O_RDONLY);
	if (fd >= 0){
		read(fd, buf, size);
	}
	close(fd);
}

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

void freeWekuaPlatform(wPlatform *plat, uint32_t nplat){
	for (uint32_t x=0; x<nplat; x++){
		free(plat[x].name);
	}
	free(plat);
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
			(*device)[x].max_work_item_sizes = (uint64_t*) malloc(s);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_ITEM_SIZES, s, (*device)[x].max_work_item_sizes, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &s);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_WORK_GROUP_SIZE, s, &(*device)[x].max_work_group_size, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 0, NULL, &s);
			clGetDeviceInfo(dev[x], CL_DEVICE_MAX_MEM_ALLOC_SIZE, s, &(*device)[x].max_size, NULL);
			clGetDeviceInfo(dev[x], CL_DEVICE_NAME, 0, NULL, &(*device)[x].nlen);
			(*device)[x].name = (uint8_t*) malloc((*device)[x].nlen);
			clGetDeviceInfo(dev[x], CL_DEVICE_NAME, (*device)[x].nlen, (*device)[x].name, NULL);
		}
		free(dev);
	}
	return ndev;
}

void freeWekuaDevice(wDevice *dev, uint32_t ndev){
	for (uint32_t x=0; x<ndev; x++){
		free(dev[x].name);
		free(dev[x].max_work_item_sizes);
		clReleaseDevice(dev[x].device);
	}
	free(dev);
}

char *getKernelData(const char *name, long int *size){
	int fd = open(name, O_RDONLY);
	if (fd < 0){
		return NULL;
	}
	*size = lseek(fd, 0, SEEK_END);
	lseek(fd, 0, SEEK_SET);
	char *cont = (char*) calloc(size[0], 1);
	if (read(fd, cont, size[0]) != size[0]){
		free(cont);
		close(fd);
		return NULL;
	}
	close(fd);
	return cont;
}

wekuaContext *createWekuaContext(wDevice *dev){
	if (dev->max_work_item_dimensions < 3){
		return NULL;
	}
	wekuaContext *context = (wekuaContext*) malloc(sizeof(wekuaContext));
	context->ctx = clCreateContext(NULL, 1, &dev->device, NULL, NULL, NULL);
	context->command_queue = clCreateCommandQueueWithProperties(context->ctx, dev->device, NULL, NULL);
	context->programs = (cl_program*) calloc(KERNEL_NUM, sizeof(cl_program));
	context->kernels = (cl_kernel*) calloc(KERNEL_NUM, sizeof(cl_kernel));
	for (uint32_t x=0; x<KERNEL_NUM; x++){
		uint64_t size;
		char *source = getKernelData(kernels[x], (long int*)&size);
		context->programs[x] = clCreateProgramWithSource(context->ctx, 1, (const char**)&source, &size, NULL);
		if (clBuildProgram(context->programs[x], 1, &dev->device, NULL, NULL, NULL) != 0){
			clGetProgramBuildInfo(context->programs[x], dev->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
			free(source);
			source = (char*) malloc(size);
			clGetProgramBuildInfo(context->programs[x], dev->device, CL_PROGRAM_BUILD_LOG, size, source, NULL);
			printf("%s\n", source);
			free(source);
			freeWekuaContext(context);
			return NULL;
		}
		context->kernels[x] = clCreateKernel(context->programs[x], ker_name[x], NULL);

		free(source);
	}
	context->max_work_group_size = dev->max_work_group_size;
	context->max_work_item_dimensions = dev->max_work_item_dimensions;
	context->max_work_item_sizes = calloc(context->max_work_item_dimensions, 8);
	memcpy(context->max_work_item_sizes, dev->max_work_item_sizes, 8*dev->max_work_item_dimensions);
	return context;
}

wekuaContext *createSomeWekuaContext(wekua_device_type type){
	wDevice **devs;
	wPlatform *plat;
	wekuaContext *ctx;
	uint32_t nplat, *ndev, ps=0, ds=0;
	nplat = getPlatforms(&plat);
	devs = (wDevice**) calloc(nplat, sizeof(wDevice*));
	ndev = (uint32_t*) calloc(nplat, 4);
	for (uint32_t p=0; p<nplat; p++){
		ndev[p] = getDevices(plat[p], &devs[p], type);
		for (uint32_t d=0; d<ndev[p]; d++){
			if (devs[p][d].compute_units*devs[p][d].clock_frequency*devs[p][d].max_work_group_size > devs[ps][ds].compute_units*devs[ps][ds].clock_frequency*devs[ps][ds].max_work_group_size){
				ps = p; ds = d;
			}else if (devs[p][d].compute_units*devs[p][d].clock_frequency*devs[p][d].max_work_group_size == devs[ps][ds].compute_units*devs[ps][ds].clock_frequency*devs[ps][ds].max_work_group_size){
				if (devs[p][d].max_size > devs[ps][ds].max_size){
					ps = p; ds = d;
				}
			}
		}
	}
	ctx = createWekuaContext(&devs[ps][ds]);
	freeWekuaPlatform(plat, nplat);
	for (uint32_t p=0; p<nplat; p++){
		freeWekuaDevice(devs[p], ndev[p]);
	}
	free(devs); free(ndev);
	return ctx;
}

void freeWekuaContext(wekuaContext *context){
	for (uint32_t x=0; x<KERNEL_NUM; x++){
		clReleaseProgram(context->programs[x]);
		clReleaseKernel(context->kernels[x]);
	}
	clReleaseContext(context->ctx);
	clFlush(context->command_queue);
	clReleaseCommandQueue(context->command_queue);
	free(context->kernels);
	free(context->programs);
	free(context->max_work_item_sizes);
	free(context);
}
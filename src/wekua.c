#include "wekua.h"
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
// #include <stdarg.h>

#define KERNEL_NUM 33

const char kernels[KERNEL_NUM][50] = {
	"/usr/lib/wekua_kernels/rand.cl",
	"/usr/lib/wekua_kernels/randuniform.cl",
	"/usr/lib/wekua_kernels/iden.cl",
	"/usr/lib/wekua_kernels/trans.cl",
	"/usr/lib/wekua_kernels/axpy.cl",
	"/usr/lib/wekua_kernels/scal.cl",
	"/usr/lib/wekua_kernels/dot.cl",
	"/usr/lib/wekua_kernels/convert.cl",
	"/usr/lib/wekua_kernels/abs.cl",
	"/usr/lib/wekua_kernels/diag.cl",
	"/usr/lib/wekua_kernels/arange.cl",
	"/usr/lib/wekua_kernels/power.cl",
	"/usr/lib/wekua_kernels/divide.cl",
	"/usr/lib/wekua_kernels/log.cl",
	"/usr/lib/wekua_kernels/sin.cl",
	"/usr/lib/wekua_kernels/sinh.cl",
	"/usr/lib/wekua_kernels/cos.cl",
	"/usr/lib/wekua_kernels/cosh.cl",
	"/usr/lib/wekua_kernels/tan.cl",
	"/usr/lib/wekua_kernels/tanh.cl",
	"/usr/lib/wekua_kernels/mul.cl",
	"/usr/lib/wekua_kernels/fill.cl",
	"/usr/lib/wekua_kernels/euleriden.cl",
	"/usr/lib/wekua_kernels/calc_dev.cl",
	"/usr/lib/wekua_kernels/aberth.cl",
	"/usr/lib/wekua_kernels/det.cl",
	"/usr/lib/wekua_kernels/gauss.cl",
	"/usr/lib/wekua_kernels/gauss2.cl",
	"/usr/lib/wekua_kernels/bias.cl",
	"/usr/lib/wekua_kernels/sigmoid.cl",
	"/usr/lib/wekua_kernels/gemm.cl",
	"/usr/lib/wekua_kernels/sum.cl",
	"/usr/lib/wekua_kernels/linear_bias_amend.cl"
};

const char ker_name[KERNEL_NUM][20] = {
	"rand", "uniform", "iden", "trans", "axpy",
	"scal", "doth", "convert",
	"absolute", "diag", "arange", "power",
	"divide", "lognatu", "sen", "senh", "cose",
	"coseh", "tg", "tgh", "mul", "fill",
	"euler_iden", "calc_dev", "aberth", "det",
	"gauss", "gauss2", "bias", "sigmoid",
	"gemm", "sum_kernel"
};

const uint32_t dtype_length[10] = {
	1, 1, // int8_t
	2, 2, // int16_t
	4, 4, // int32_t
	8, 8, // int64_t
	sizeof(float),
	sizeof(double)
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
			wekuaPlatformFromclPlatform(plat[x], &(*platform)[x]);
		}
		free(plat);
	}
	return nplat;
}

void wekuaPlatformFromclPlatform(cl_platform_id platform, wPlatform *plat){
	plat->platform = platform;

	clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &plat->nlen);
	plat->name = (uint8_t*) calloc(plat->nlen, 1);

	clGetPlatformInfo(platform, CL_PLATFORM_NAME, plat->nlen, plat->name, NULL);
}

void freeWekuaPlatform(wPlatform *plat, uint32_t nplat){
	for (uint32_t x=0; x<nplat; x++){
		free(plat[x].name);
	}
	free(plat);
}

uint32_t getDevices(wPlatform platform , wDevice **device, cl_device_type type){
	uint32_t ndev;
	clGetDeviceIDs(platform.platform, type, 0, NULL, &ndev);
	if (ndev > 0){
		cl_device_id *dev = (cl_device_id*) calloc(ndev, sizeof(cl_device_id));
		*device = (wDevice*) calloc(ndev, sizeof(wDevice));
		clGetDeviceIDs(platform.platform, type, ndev, dev, NULL);
		for (uint32_t x=0; x<ndev; x++){
			wekuaDeviceFromclDevice(dev[x], &(*device)[x]);
		}
		free(dev);
	}
	return ndev;
}

void wekuaDeviceFromclDevice(cl_device_id dev, wDevice *wdev){
	wdev->device = dev;

	clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &wdev->type, NULL);

	clGetDeviceInfo(dev, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, 4, &wdev->compute_units, NULL);

	clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, 4, &wdev->clock_frequency, NULL);

	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 4, &wdev->max_work_item_dimensions, NULL);			

	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, 8, &wdev->max_work_group_size, NULL);

	clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &wdev->max_global_size, NULL);

	clGetDeviceInfo(dev, CL_DEVICE_PARTITION_PROPERTIES, 3*sizeof(cl_device_partition_property), wdev->propie, NULL);

	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, 4, wdev->vector_width, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, 4, &wdev->vector_width[1], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, 4, &wdev->vector_width[2], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, 4, &wdev->vector_width[3], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, 4, &wdev->vector_width[4], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, 4, &wdev->vector_width[5], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, 4, &wdev->vector_width[6], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, 4, &wdev->vector_width[7], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, 4, &wdev->vector_width[8], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, 4, &wdev->vector_width[9], NULL);

	clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &wdev->nlen);
	wdev->name = (uint8_t*) malloc(wdev->nlen);

	clGetDeviceInfo(dev, CL_DEVICE_NAME, wdev->nlen, wdev->name, NULL);
}

void freeWekuaDevice(wDevice *dev, uint32_t ndev){
	if (dev == NULL){
		return;
	}

	for (uint32_t x=0; x<ndev; x++){
		free(dev[x].name);
		if (dev[x].device != NULL) clReleaseDevice(dev[x].device);
	}
	free(dev);
}

char *getKernelData(const char *name, long *size){
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

wekuaContext createWekuaContext(wDevice *dev){
	if (dev == NULL){
		return NULL;
	}else if (dev->max_work_item_dimensions < 3) return NULL;

	int ret;

	wekuaContext context = (wekuaContext) calloc(1, sizeof(struct _wk_ctx));
	if (context == NULL) return NULL;

	context->ctx = clCreateContext(NULL, 1, &dev->device, NULL, NULL, &ret);
	if (ret != CL_SUCCESS){
		free(context);
		return NULL;
	}

	context->command_queue = clCreateCommandQueueWithProperties(context->ctx, dev->device, NULL, NULL);

	context->programs = (cl_program*) calloc(KERNEL_NUM*10, sizeof(cl_program));
	context->kernels = (cl_kernel*) calloc(KERNEL_NUM*10, sizeof(cl_kernel));

	memcpy(context->vector_width, dev->vector_width, 40);

	context->dev = dev->device;
	dev->device = NULL;

	context->dtype_length = dtype_length;
	context->max_work_group_size = dev->max_work_group_size;
	context->compute_units = dev->compute_units;

	// context->vector_width[9] = 1;

	return context;
}

wekuaContext createSomeWekuaContext(cl_device_type type){
	wDevice **devs;
	wPlatform *plat;
	wekuaContext ctx;
	uint32_t nplat, *ndev, ps=0, ds=0;
	nplat = getPlatforms(&plat);
	devs = (wDevice**) calloc(nplat, sizeof(wDevice*));
	ndev = (uint32_t*) calloc(nplat, 4);

	for (uint32_t p=0; p<nplat; p++){
		ndev[p] = getDevices(plat[p], &devs[p], type);
		for (uint32_t d=0; d<ndev[p]; d++){
			if (devs[ps] == NULL){
				ps = p;
			}

			if (devs[p][d].compute_units*devs[p][d].clock_frequency*devs[p][d].max_work_group_size > devs[ps][ds].compute_units*devs[ps][ds].clock_frequency*devs[ps][ds].max_work_group_size){
				ps = p; ds = d;
			}else if (devs[p][d].compute_units*devs[p][d].clock_frequency*devs[p][d].max_work_group_size == devs[ps][ds].compute_units*devs[ps][ds].clock_frequency*devs[ps][ds].max_work_group_size){
				if (devs[p][d].max_global_size > devs[ps][ds].max_global_size){
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

uint8_t compileKernel(wekuaContext ctx, uint8_t id, uint8_t dtype){
	if (ctx->kernels[id*10+dtype] != NULL){
		return 0;
	}

	char args[30], *source, *error;

	sprintf(args, "-Dwidth=%d -Ddtype=%d", ctx->vector_width[dtype], dtype);

	uint64_t size;
	int ret;

	source = getKernelData(kernels[id], (long*)&size);

	ctx->programs[id*10+dtype] = clCreateProgramWithSource(ctx->ctx, 1, (const char**)&source, &size, &ret);
	if (ret != CL_SUCCESS) return 1;

	ret = clBuildProgram(ctx->programs[id*10+dtype], 1, &ctx->dev, args, NULL, NULL);
	if (ret != CL_SUCCESS){
		clGetProgramBuildInfo(ctx->programs[id*10+dtype], ctx->dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		error = (char*) malloc(size);
		clGetProgramBuildInfo(ctx->programs[id*10+dtype], ctx->dev, CL_PROGRAM_BUILD_LOG, size, error, NULL);
		printf("%s\nSize: %ld\n", error, size);
		free(error);
		
		printf("Return Code: %d\nKernel: %s\n", ret, ker_name[id]);
		
		clReleaseProgram(ctx->programs[id*10+dtype]);
		ctx->programs[id*10+dtype] = NULL;
		free(source);
		return 1;
	}
	free(source);

	ctx->kernels[id*10+dtype] = clCreateKernel(ctx->programs[id*10+dtype], ker_name[id], &ret);
	if (ret != CL_SUCCESS) return 1;

	return 0;
}

void freeWekuaContext(wekuaContext context){
	if (context == NULL) return;

	if (context->kernels != NULL){
		for (uint32_t x=0; x<KERNEL_NUM*10; x++){
			if (context->kernels[x] != NULL) clReleaseKernel(context->kernels[x]);
		}
	}

	if (context->programs != NULL){
		for (uint32_t x=0; x<KERNEL_NUM*10; x++){
			if (context->programs[x] != NULL) clReleaseProgram(context->programs[x]);
		}
	}

	if (context->command_queue != NULL){
		clFinish(context->command_queue);
		clReleaseCommandQueue(context->command_queue);
	}

	if (context->ctx != NULL) clReleaseContext(context->ctx);
	if (context->kernels != NULL) free(context->kernels);
	if (context->programs != NULL) free(context->programs);
	free(context);
}
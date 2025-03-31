#ifndef __STDC_WANT_LIB_EXT2__
#define __STDC_WANT_LIB_EXT2__ 1
#endif

#include "../headers/wekua.h"
#include "../headers/matrix.h"

#include <unistd.h>

#define _GNU_SOURCE
#include <stdio.h>

#include <fcntl.h>

#define WEKUA_KERNEL_NUM 49
#define KERNEL_COL 10*WEKUA_KERNEL_NUM

static const char kernels[WEKUA_KERNEL_NUM][50] = {
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
	"/usr/lib/wekua_kernels/linear_bias_step.cl",
	"/usr/lib/wekua_kernels/sqrt.cl",
	"/usr/lib/wekua_kernels/adagrad.cl",
	"/usr/lib/wekua_kernels/gdm.cl",
	"/usr/lib/wekua_kernels/rmsprop.cl",
	"/usr/lib/wekua_kernels/adadelta.cl",
	"/usr/lib/wekua_kernels/relu.cl",
	"/usr/lib/wekua_kernels/relu_dev.cl",
	"/usr/lib/wekua_kernels/leakyrelu.cl",
	"/usr/lib/wekua_kernels/leakyrelu_dev.cl",
	"/usr/lib/wekua_kernels/mse.cl",
	"/usr/lib/wekua_kernels/sigmoid_dev.cl",
	"/usr/lib/wekua_kernels/tanh_dev.cl",
	"/usr/lib/wekua_kernels/adam.cl",
	"/usr/lib/wekua_kernels/scalar_add.cl",
	"/usr/lib/wekua_kernels/l1_regu.cl",
	"/usr/lib/wekua_kernels/regularization.cl"
};

static const char ker_name[WEKUA_KERNEL_NUM][20] = {
	"rand", "uniform", "iden", "trans", "axpy",
	"scal", "doth", "convert",
	"absolute", "diag", "arange", "power",
	"divide", "lognatu", "sen", "senh", "cose",
	"coseh", "tg", "tgh", "mul", "fill",
	"euler_iden", "calc_dev", "aberth", "det",
	"gauss", "gauss2", "bias", "sigmoid",
	"gemm", "sum_kernel", "linear_bias_step",
	"sqrt_kernel", "adagrad", "gdm",
	"rmsprop", "adadelta", "relu", "relu_dev",
	"leakyrelu", "leakyrelu_dev", "mse",
	"sigmoid_dev", "tanh_dev", "adam",
	"scalar_add", "l1regu", "regularization"
};

const uint32_t dtype_length[10] = {
	1, 1, // int8_t
	2, 2, // int16_t
	4, 4, // int32_t
	8, 8, // int64_t
	sizeof(float),
	sizeof(double)
};

int getRandomBuffer(void *buf, size_t size){
	int fd = open("/dev/urandom", O_RDONLY);
	if (fd >= 0){
		size_t bytes_read = 0;
		while (bytes_read < size) {
			ssize_t ret = read(fd, buf + bytes_read, (size - bytes_read));
			if (ret < 0) {
				close(fd);
				return -1;
			}
			bytes_read += (size_t)ret;
		}
	}
	close(fd);
	return 0;
}

uint32_t getPlatforms(wPlatform **platform){
	uint32_t nplat;
	clGetPlatformIDs(0, NULL, &nplat);
	if (nplat > 0){
		cl_platform_id *plat = (cl_platform_id*) calloc(nplat, sizeof(cl_platform_id));
		if (plat == NULL) return 0;

		*platform = (wPlatform*) calloc(nplat, sizeof(wPlatform));
		if (*platform == NULL) {
			free(plat);
			return 0;
		}
		clGetPlatformIDs(nplat, plat, NULL);
		for (uint32_t x=0; x<nplat; x++){
			wekuaPlatformFromclPlatform(plat[x], &(*platform)[x]);
		}
		free(plat);
	}else{
		*platform = NULL;
	}
	return nplat;
}

void wekuaPlatformFromclPlatform(cl_platform_id platform, wPlatform *plat){
	if (plat == NULL) return;

	plat->platform = platform;

	clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &plat->nlen);
	plat->name = (uint8_t*) calloc(plat->nlen, 1);

	clGetPlatformInfo(platform, CL_PLATFORM_NAME, plat->nlen, plat->name, NULL);
}

void freeWekuaPlatform(wPlatform *plat, uint32_t nplat){
	for (uint32_t x=0; x<nplat; x++) free(plat[x].name);
	free(plat);
}

uint32_t getDevices(wPlatform *platform , wDevice **device, cl_device_type type){
	if (platform == NULL) return 0;
	if (device == NULL) return 0;

	uint32_t ndev;
	clGetDeviceIDs(platform->platform, type, 0, NULL, &ndev);
	if (ndev > 0){
		cl_device_id *dev = (cl_device_id*) calloc(ndev, sizeof(cl_device_id));
		if (dev == NULL) return 0;

		*device = (wDevice*) calloc(ndev, sizeof(wDevice));
		if (*device == NULL) {
			free(dev);
			return 0;
		}

		clGetDeviceIDs(platform->platform, type, ndev, dev, NULL);
		for (uint32_t x=0; x<ndev; x++){
			(*device)[x].platform = platform;
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
	clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &wdev->local_mem_type, NULL);

	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, 4, wdev->vector_width, NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, 4, &wdev->vector_width[1], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, 4, &wdev->vector_width[2], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, 4, &wdev->vector_width[3], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, 4, &wdev->vector_width[4], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, 4, &wdev->vector_width[5], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, 4, &wdev->vector_width[6], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, 4, &wdev->vector_width[7], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, 4, &wdev->vector_width[8], NULL);
	clGetDeviceInfo(dev, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, 4, &wdev->vector_width[9], NULL);

	clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &wdev->nlen);
	wdev->name = (uint8_t*) malloc(wdev->nlen);

	clGetDeviceInfo(dev, CL_DEVICE_NAME, wdev->nlen, wdev->name, NULL);
}

void freeWekuaDevice(wDevice *dev, uint32_t ndev){
	if (dev == NULL) return;

	for (uint32_t x=0; x<ndev; x++){
		free(dev[x].name);
		if (dev[x].device != NULL) clReleaseDevice(dev[x].device);
	}
	free(dev);
}

static char *getKernelData(const char *name, size_t *size){
	const int fd = open(name, O_RDONLY);
	if (fd < 0){
		return NULL;
	}

	char *content = NULL;
	ssize_t lseek_ret = lseek(fd, 0, SEEK_END);
	if (lseek_ret <= 0) {
		goto getKernelData_finish;
	}
	*size = (size_t)lseek_ret;

	lseek_ret = lseek(fd, 0, SEEK_SET);
	if (lseek_ret < 0) {
		goto getKernelData_finish;
	}

	content = (char*) calloc(size[0], 1);
	const ssize_t data_read = read(fd, content, size[0]);
	if (data_read != ((ssize_t)size[0])){
		goto getKernelData_error;
	}

	goto getKernelData_finish;

getKernelData_error:
	if (content != NULL) {
		free(content);
		content = NULL;
	}

getKernelData_finish:
	close(fd);
	return content;
}


struct w_matrix_free_arg {
	uint8_t service;
	pthread_mutex_t lock;
	wfifo fifo;
};

static void *wekua_matrix_free_worker(void *arg){
	struct w_matrix_free_arg *data = arg;
	pthread_mutex_t *lock = &data->lock;
	wfifo fifo = data->fifo;
	uint8_t run;

	struct _w_obj {
		int (*free)(void *);
	} *obj;

	while (1){
		pthread_mutex_lock(lock);
		run = data->service|wekuaFIFOisnotEmpty(fifo);
		pthread_mutex_unlock(lock);
		if (run){
			obj = wekuaFIFOGet(fifo);
			if (obj){
				if (obj->free(obj) != CL_SUCCESS) wekuaFIFOPut(fifo, obj);
			}else if (wekuaFIFOisEmpty(fifo)) break;
		}else break;
	}
	return NULL;
}

wekuaContext createWekuaContextsFromOpenCLContext(cl_context ctx, wDevice *dev, uint8_t use_vectors, uint8_t alloc_host_mem){
	if (dev == NULL){
		return NULL;
	}else if (dev->max_work_item_dimensions < 3) return NULL;

	wekuaContext context = (wekuaContext) calloc(1, sizeof(struct _wk_ctx));
	if (context == NULL) return NULL;

	context->ctx = ctx;

	context->command_queue = clCreateCommandQueueWithProperties(context->ctx, dev->device, NULL, NULL);

	context->programs = (cl_program*) calloc(2*KERNEL_COL, sizeof(cl_program));
	context->kernels = (cl_kernel*) calloc(2*KERNEL_COL, sizeof(cl_kernel));

	if (use_vectors){
		memcpy(context->vector_width, dev->vector_width, 40);
	}else{
		for (uint8_t x=0; x<10; x++) context->vector_width[x] = 1;
	}

	context->dev = dev->device;
	dev->device = NULL;

	context->dtype_length = dtype_length;
	context->max_work_group_size = dev->max_work_group_size;
	context->compute_units = dev->compute_units;
	context->local_mem_type = dev->local_mem_type;
	if (alloc_host_mem) context->flags = CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR;
	else context->flags = CL_MEM_READ_WRITE;

	wfifo garbage_queue = wekuaAllocFIFO();
	struct w_matrix_free_arg *data = calloc(1, sizeof(struct w_matrix_free_arg));
	if (data == NULL) {
		wekuaFreeFIFO(garbage_queue);
		freeWekuaContext(context);
		return NULL;
	}

	pthread_mutex_init(&data->lock, NULL);
	data->fifo = garbage_queue;
	data->service = 1;
	context->garbage_queue = garbage_queue;
	context->garbage_collector_arg = data;
	pthread_create(&context->garbage_collector, NULL, &wekua_matrix_free_worker, data);

	return context;
}

wekuaContext createWekuaContext(wDevice *dev, uint8_t use_vectors, uint8_t alloc_host_mem){
	if (dev == NULL){
		return NULL;
	}else if (dev->max_work_item_dimensions < 3) return NULL;

	int ret;
	cl_context ctx = clCreateContext(NULL, 1, &dev->device, NULL, NULL, &ret);
	if (ret != CL_SUCCESS){
		return NULL;
	}
	wekuaContext context = createWekuaContextsFromOpenCLContext(ctx, dev, use_vectors, alloc_host_mem);
	return context;
}

wekuaContext createSomeWekuaContext(cl_device_type type, uint8_t use_vectors, uint8_t alloc_host_mem){
	wDevice **devs;
	wPlatform *plat;
	wekuaContext ctx;
	uint32_t nplat, *ndev, ps=0, ds=0;
	nplat = getPlatforms(&plat);
	if (nplat == 0) {
		return NULL;
	}

	devs = (wDevice**) calloc(nplat, sizeof(wDevice*));
	if (devs == NULL) {
		freeWekuaPlatform(plat, nplat);
		return NULL;
	}
	ndev = (uint32_t*) calloc(nplat, 4);
	if (ndev == NULL) {
		free(devs);
		freeWekuaPlatform(plat, nplat);
		return NULL;
	}

	for (uint32_t p=0; p<nplat; p++){
		ndev[p] = getDevices(&plat[p], &devs[p], type);
		for (uint32_t d=0; d<ndev[p]; d++){
			if (devs[ps] == NULL){
				ps = p;
			}
			if (devs[p] == NULL) continue;

			if (devs[p][d].compute_units*devs[p][d].clock_frequency*devs[p][d].max_work_group_size > devs[ps][ds].compute_units*devs[ps][ds].clock_frequency*devs[ps][ds].max_work_group_size){
				ps = p; ds = d;
			}else if (devs[p][d].compute_units*devs[p][d].clock_frequency*devs[p][d].max_work_group_size == devs[ps][ds].compute_units*devs[ps][ds].clock_frequency*devs[ps][ds].max_work_group_size){
				if (devs[p][d].max_global_size > devs[ps][ds].max_global_size){
					ps = p; ds = d;
				}
			}
		}
	}
	ctx = createWekuaContext(&devs[ps][ds], use_vectors, alloc_host_mem);
	freeWekuaPlatform(plat, nplat);
	for (uint32_t p=0; p<nplat; p++){
		freeWekuaDevice(devs[p], ndev[p]);
	}
	free(devs); free(ndev);
	return ctx;
}

cl_kernel compileCustomKernel(wekuaContext ctx, const char *filename, const char *kernel_name, char *args, cl_program *program){
	cl_kernel kernel;
	char *source, *error;

	uint64_t size;
	int ret;

	source = getKernelData(filename, (size_t*)&size);

	program[0] = clCreateProgramWithSource(ctx->ctx, 1, (const char**)&source, &size, &ret);
	if (ret != CL_SUCCESS) return NULL;

	ret = clBuildProgram(program[0], 1, &ctx->dev, args, NULL, NULL);
	if (ret != CL_SUCCESS){
		clGetProgramBuildInfo(program[0], ctx->dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		error = (char*) malloc(size);
		clGetProgramBuildInfo(program[0], ctx->dev, CL_PROGRAM_BUILD_LOG, size, error, NULL);
		printf("%s\nSize: %ld\n", error, size);
		free(error);
		
		printf("Return Code: %d\nKernel: %s\n", ret, kernel_name);
		
		clReleaseProgram(program[0]);
		program[0] = NULL;
		free(source);
		return NULL;
	}
	free(source);

	kernel = clCreateKernel(program[0], kernel_name, &ret);
	if (ret != CL_SUCCESS){
		clReleaseProgram(program[0]);
		program[0] = NULL;
		return NULL;
	}

	return kernel;
}

cl_kernel compileKernel(wekuaContext ctx, uint8_t id, uint8_t dtype, uint8_t com){
	uint32_t pos = (uint32_t)(com*KERNEL_COL + id*10 + dtype);
	cl_kernel kernel = ctx->kernels[pos];
	if (kernel != NULL) return kernel;

	char *args = NULL;
	int ret = asprintf(
		&args, "-Dwk_width=%d -Ddtype=%d -Dmem_type=%d -Dcom=%d",
		ctx->vector_width[dtype], dtype, ctx->local_mem_type, com
	);
	if (ret < 0) return NULL;

	kernel = compileCustomKernel(ctx, kernels[id], ker_name[id], args, &ctx->programs[pos]);
	ctx->kernels[pos] = kernel;
	free(args);

	return kernel;
}

void freeWekuaContext(wekuaContext context){
	if (context == NULL) return;

	struct w_matrix_free_arg *data = context->garbage_collector_arg;
	if (data != NULL) {
		wfifo fifo = data->fifo;
		pthread_mutex_lock(&data->lock);
		data->service = 0;
		pthread_mutex_unlock(&data->lock);
		wekuaFIFOPut(fifo, NULL);
		pthread_join(context->garbage_collector, NULL);

		pthread_mutex_destroy(&data->lock);
		wekuaFreeFIFO(fifo);
		free(data);
	}

	if (context->kernels){
		for (uint32_t x=0; x<KERNEL_COL*2; x++){
			if (context->kernels[x]) clReleaseKernel(context->kernels[x]);
		}
	}

	if (context->programs){
		for (uint32_t x=0; x<KERNEL_COL*2; x++){
			if (context->programs[x]) clReleaseProgram(context->programs[x]);
		}
	}

	if (context->command_queue){
		clFinish(context->command_queue);
		clReleaseCommandQueue(context->command_queue);
	}

	if (context->ctx) clReleaseContext(context->ctx);
	if (context->kernels) free(context->kernels);
	if (context->programs) free(context->programs);
	free(context);
}

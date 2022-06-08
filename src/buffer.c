#include "buffer.h"

#define CL_MEM_LOCATION_HOST_NV                     (1 << 0)
#define CL_MEM_PINNED_NV                            (1 << 1)

#define EXTENSIONS_NUMBER 2

static const char extensions_name[EXTENSIONS_NUMBER][20] = {
	"clCreateBufferNV",

	// ...
	"clCreateBuffer"
};

typedef cl_mem (*cbf_array)(cl_context, void *, uint64_t, int*);
static const cbf_array functions_ptr[EXTENSIONS_NUMBER] = {
	&createNVIDIABuffer,

	// ...
	&createNormalBuffer
};


void getBuffersFunctions(wekuaContext ctx, cl_platform_id platform){
	uint8_t x = 0;
	void *ptr = NULL;
	for (; x < (EXTENSIONS_NUMBER - 1); x++){
		if ((ptr = clGetExtensionFunctionAddressForPlatform(platform, extensions_name[x]))) break;
	}

	ctx->alloc_buffer_function = ptr;
	ctx->create_new_buffer = functions_ptr[x];
}

void getBuffersFunctionsForPlatform(wekuaPlatformContext ctx, cl_platform_id platform){
	uint8_t x = 0;
	void *ptr = NULL;
	for (; x < (EXTENSIONS_NUMBER - 1); x++){
		if ((ptr = clGetExtensionFunctionAddressForPlatform(platform, extensions_name[x]))) break;
	}

	ctx->alloc_buffer_function = ptr;
	ctx->create_new_buffer = functions_ptr[x];
}

cl_mem createNormalBuffer(cl_context ctx, void *func, uint64_t size, int *ret){
	return clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, size, NULL, ret);
}

cl_mem createNVIDIABuffer(cl_context ctx, void *func, uint64_t size, int *ret){
	cl_mem (*clCreateBufferNV)(cl_context,cl_mem_flags, cl_bitfield, size_t, void*, cl_int*) = func;
	return clCreateBufferNV(ctx, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, CL_MEM_PINNED_NV|CL_MEM_LOCATION_HOST_NV, size, NULL, ret);
}

cl_mem wekuaCreateBuffer(wekuaContext ctx, uint64_t size, int *ret){
	return ctx->create_new_buffer(ctx->ctx, ctx->alloc_buffer_function, size, ret);
}
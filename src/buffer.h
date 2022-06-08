#ifndef BUFFER_H
#define BUFFER_H

#include "../headers/wekua.h"


cl_mem createNormalBuffer(cl_context ctx, void *func, uint64_t size, int *ret);
cl_mem createNVIDIABuffer(cl_context ctx, void *func, uint64_t size, int *ret); // NVIDIA buffer function

cl_mem wekuaCreateBuffer(wekuaContext ctx, uint64_t size, int *ret);

void getBuffersFunctions(wekuaContext ctx, cl_platform_id platform);
void getBuffersFunctionsForPlatform(wekuaPlatformContext ctx, cl_platform_id platform);


#endif
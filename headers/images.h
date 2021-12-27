#ifndef IMAGES_H
#define IMAGES_H

#include "matrix.h"

#define WEKUA_IMAGE_FMT_PNG 0
#define WEKUA_IMAGE_FMT_JPEG 1

#define WEKUA_IMAGE_COLOR_FMT_UNKOWN 0
#define WEKUA_IMAGE_COLOR_FMT_RGBA8 1 // Red, Green, Blue, Alpha - 8 bits
#define WEKUA_IMAGE_COLOR_FMT_RGBA16 2 // Red, Green, Blue, Alpha - 16 bits
#define WEKUA_IMAGE_COLOR_FMT_RGB8 3 // Red, Green, Blue - 8 bits
#define WEKUA_IMAGE_COLOR_FMT_G8 4 // Greyscale - 8 bits
#define WEKUA_IMAGE_COLOR_FMT_GA16 5 // Greyscale with alpha - 16 bits
#define WEKUA_IMAGE_COLOR_FMT_GA8 6 // Greyscale with alpha - 8 bits

typedef struct _w_images {
	wmatrix buffer;
	uint8_t color_fmt; // Color format
	uint64_t num; // Images num
	uint64_t layers; // Layers num
} *wimages;

#ifdef __cplusplus
extern "C" {
#endif

wimages wekuaAllocCustomImage(wekuaContext ctx, uint64_t num, uint64_t layers);
wimages wekuaAllocRGBA8Images(wekuaContext ctx, uint64_t num);
wimages wekuaAllocRGBA16Images(wekuaContext ctx, uint64_t num);
wimages wekuaAllocRGB8Images(wekuaContext ctx, uint64_t num);
wimages wekuaAllocG8Images(wekuaContext ctx, uint64_t num);
wimages wekuaAllocGA16Images(wekuaContext ctx, uint64_t num);
wimages wekuaAllocGA8Images(wekuaContext ctx, uint64_t num);

void loadImageFileToWekuaImage(const char *filename, wimages imgs, uint64_t offset);
void saveImageFileToWekuaImage(const char *filename, wimages imgs, uint64_t offset, uint8_t color_fmt);

void wekuaFreeImages(wimages imgs, uint32_t nw, cl_event *be);

#ifdef __cplusplus
}
#endif

#endif
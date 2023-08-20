#define _GNU_SOURCE
#include <stdio.h>
#include <assert.h>
#include <sys/param.h>

#include "tensor.h"

#define PRINT_REAL_VALUES \
switch (dtype) { \
	case WEKUA_DTYPE_UINT8:  \
		printf("%s%5u", padding, ((uint8_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_UINT16: \
		printf("%s%7u", padding, ((uint16_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_UINT32: \
		printf("%s%12u", padding, ((uint32_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_UINT64: \
		printf("%s%22lu", padding, ((uint64_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_INT8: \
		printf("%s%5i", padding, ((int8_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_INT16: \
		printf("%s%7i", padding, ((int16_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_INT32: \
		printf("%s%12i", padding, ((int32_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_INT64: \
		printf("%s%22li", padding, ((int64_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_FLOAT32: \
		printf("%s%14.5e", padding, ((float *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_FLOAT64: \
		printf("%s%14.5e", padding, ((double *)tmp_buf)[_off[0]++]); \
		break; \
} \

#define PRINT_COMPLEX_VALUE(dtype, dtype2, fmt, fmt1, fmt2, fmt3) \
static void print_##dtype##_value(const void *data, uint64_t *off) { \
	const dtype2 real = (dtype2) ((dtype*)data)[off[0]++]; \
	const dtype2 imag = (dtype2) ((dtype*)data)[off[0]++]; \
	char message[30]; \
	if (real != 0 && imag != 0) { \
		sprintf(message, fmt1, real, imag); \
	}else if (imag != 0) { \
		sprintf(message, fmt2, real); \
	}else{ \
		sprintf(message, fmt3, real); \
	} \
	printf(fmt, message); \
} \

PRINT_COMPLEX_VALUE(uint8_t, uint8_t, "%10s", "%u+%uj", "%uj", "%u")
PRINT_COMPLEX_VALUE(uint16_t, uint16_t, "%14s", "%u+%uj", "%uj", "%u")
PRINT_COMPLEX_VALUE(uint32_t, double, "%24s", "%.2e+%.2ej", "%.5ej", "%.5e")
PRINT_COMPLEX_VALUE(uint64_t, double, "%24s", "%.2e+%.2ej", "%.5ej", "%.5e")
PRINT_COMPLEX_VALUE(int8_t, int8_t, "%10s", "%d+%dj", "%dj", "%d")
PRINT_COMPLEX_VALUE(int16_t, int16_t, "%14s", "%d+%dj", "%dj", "%d")
PRINT_COMPLEX_VALUE(int32_t, double, "%24s", "%.2e+%.2ej", "%.5ej", "%.5e")
PRINT_COMPLEX_VALUE(int64_t, double, "%24s", "%.2e+%.2ej", "%.5ej", "%.5e")
PRINT_COMPLEX_VALUE(float, float, "%24s", "%.2e+%.2ej", "%.5ej", "%.5e")
PRINT_COMPLEX_VALUE(double, double, "%24s", "%.2e+%.2ej", "%.5ej", "%.5e")

#define PRINT_COMPLEX_VALUES \
switch (dtype) { \
	case WEKUA_DTYPE_UINT8:  \
		print_uint8_t_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_UINT16: \
		print_uint16_t_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_UINT32: \
		print_uint32_t_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_UINT64: \
		print_uint64_t_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_INT8: \
		print_int8_t_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_INT16: \
		print_int16_t_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_INT32: \
		print_int32_t_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_INT64: \
		print_int64_t_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_FLOAT32: \
		print_float_value(tmp_buf, _off); \
		break; \
	case WEKUA_DTYPE_FLOAT64: \
		print_double_value(tmp_buf, _off); \
		break; \
} \

#define print_elements(func_name, label_name) \
	static void func_name( \
		const char *padding, const uint64_t r, uint64_t limit_r, \
		const uint64_t c, const uint64_t limit_c, const void *tmp_buf, \
		uint64_t *_off, const uint8_t dtype \
	){ \
		uint64_t y = 0; \
		for (uint8_t i=0; i<2 && limit_r; i++){ \
			for (; y<limit_r; y++){ \
				printf("%s", padding); \
				uint64_t x = 0; \
				for (; x<limit_c; x++){ \
					label_name \
				} \
				if (c > 8) printf("%10s", "..."); \
				x = MAX((c - x), limit_c); \
				for (; x<c; x++){ \
					label_name \
				} \
				printf("\n"); \
			} \
			if (r > 8 && !i) printf("%s%10s\n", padding, "..."); \
			y = MAX((r - y), limit_r); \
			limit_r = r; \
		} \
	} \

// static const char formats[][6] = {
// 	"5u",
// 	"7u",
// 	"12u",
// 	"22lu",
// 	"5i",
// 	"7i",
// 	"12i",
// 	"22li",
// 	"14.5e",
// 	"14.5e"
// };

static const char dtype_text[][15] = {
	"uint8", "complex_uint8",
	"uint16", "complex_uint16",
	"uint32", "complex_uint32",
	"uint64", "complex_uint64",
	"int8", "complex_int8",
	"int16", "complex_int16",
	"int32", "complex_int32",
	"int64", "complex_int64",
	"float32", "complex_float32",
	"float64", "complex_float64"
};

static void get_data_from_device(
	cl_command_queue cmd, cl_mem buf, uint64_t buf_offset, const uint64_t stride, uint64_t *cache_offset, void *cache,
	uint64_t y, const uint64_t limit_r, const uint64_t c, const uint64_t limit_c, const uint32_t dtype_length,
	const uint8_t com, cl_event *events, uint32_t *nevents
){
	for (; y<limit_r; y++){
		uint64_t len = dtype_length*limit_c*(com + 1);
		int ret = clEnqueueReadBuffer(cmd, buf, CL_FALSE, buf_offset, len, cache + cache_offset[0], 0, NULL, &events[nevents[0]++]);
		assert(ret == CL_SUCCESS);

		cache_offset[0] += len;
		if (c > 4){
			uint64_t offset = c - 4;
			len = MIN(offset, 4) * dtype_length * (com + 1);
			ret = clEnqueueReadBuffer(cmd, buf, CL_FALSE, buf_offset + offset*(1 + com), len, cache + cache_offset[0], 0, NULL, &events[nevents[0]++]);
			assert(ret == CL_SUCCESS);

			cache_offset[0] += len;
		}

		buf_offset += stride;
	}
}

print_elements(print_real_elements, PRINT_REAL_VALUES)
print_elements(print_complex_elements, PRINT_COMPLEX_VALUES)

static void print_matrix(
	wtensor tensor, const char *padding, const char ndim, const uint64_t *shape, const uint64_t stride,
	const uint64_t offset, void *cache
){
	const uint64_t r = (ndim == 1) ? 1 : shape[0];
	const uint64_t c = shape[1];

	cl_mem buf = tensor->buffer;
	wekuaContext ctx = tensor->ctx;
	const uint8_t dtype = tensor->dtype;
	const uint8_t com = tensor->com;

	const uint32_t dtype_length = ctx->dtype_length[dtype];

	cl_event events[16];
	uint32_t nevents = 0;
	cl_command_queue cmd = ctx->command_queue;

	const uint64_t limit_r = MIN(r, 4);
	const uint64_t limit_c = MIN(c, 4);
	uint64_t cache_offset = 0;
	uint64_t y = 0;

	get_data_from_device(cmd, buf, offset, stride, &cache_offset, cache, y, limit_r, c, limit_c, dtype_length, com, events, &nevents);
	if (r > 4){
		y = MAX((r - 4), 4);
		get_data_from_device(cmd, buf, offset, stride, &cache_offset, cache, y, r, c, limit_c, dtype_length, com, events, &nevents);
	}

	cache_offset = 0;

	wait_for_and_release_cl_events(events, nevents);

	if (com) print_complex_elements(padding, r, limit_r, c, limit_c, cache, &cache_offset, dtype);
	else print_real_elements(padding, r, limit_r, c, limit_c, cache, &cache_offset, dtype);
}

static void print_tensor(
	wtensor tensor, char *padding, const uint64_t ndim, uint64_t *shape, const uint64_t *strides,
	const uint64_t offset, void *cache
){
	printf("%s[\n", padding);

	padding -= 2;

	const uint64_t stride = strides[0];

	if (ndim > 2){
		uint64_t len = shape[0], x = 0;
		uint64_t y = MIN(len, 3);
		
		for (; x < y; x++){
			print_tensor(tensor, padding, ndim - 1, shape + 1, strides + 1, offset + x * stride, cache);
			printf("\n");
		}

		if (len > 6) printf("%s%10s\n", padding, "...");
		if (y == 3) y = MAX((len - 3), x);

		for (; x<len; x++){
			print_tensor(tensor, padding, ndim - 1, shape + 1, strides + 1, offset + x * stride, cache);
			printf("\n");
		}
	}else{
		print_matrix(tensor, padding, ndim, shape, (ndim == 1) ? 1 : stride, offset, cache);
	}

	printf("%s]", padding + 2);
}

void wekuaTensorPrint(wtensor tensor, uint32_t nw, cl_event *events){
	if (!tensor) return;

	const uint64_t ndim = tensor->ndim;
	const uint64_t padding_size = ndim*2 + 1;
	char *padding = (char*) calloc(padding_size, sizeof(char));
	
	for (uint64_t i=0; i<padding_size; i++) padding[i] = ' ';
	padding[padding_size - 1] = 0;

	const uint8_t dtype = tensor->dtype;
	const uint8_t com = tensor->com;
	uint64_t *shape = tensor->shape;

	void *cache = calloc(64*(com + 1), tensor->ctx->dtype_length[dtype]);

	int ret = clWaitForEvents(nw, events);
	assert(ret == CL_SUCCESS || ret == CL_INVALID_VALUE);

	printf("wtensor(\n");

	print_tensor(tensor, padding + padding_size - 1, ndim, shape, tensor->strides, 0, cache);

	printf(", shape=(%lu", shape[0]);
	for (uint64_t x=1; x<ndim; x++) printf(",%lu", shape[x]);
	printf("), dtype=%s)\n", dtype_text[2*dtype + com]);

	free(cache);
	free(padding);
}

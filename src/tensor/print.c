#define _GNU_SOURCE
#include <stdio.h>
#include <assert.h>
#include <sys/param.h>

#include "tensor.h"

#define PRINT_REAL_VALUES \
switch (dtype) { \
	case WEKUA_DTYPE_UINT8:  \
		printf("%s%5u \n", padding, ((uint8_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_UINT16: \
		printf("%s%7u \n", padding, ((uint16_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_UINT32: \
		printf("%s%12u \n", padding, ((uint32_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_UINT64: \
		printf("%s%22lu \n", padding, ((uint64_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_INT8: \
		printf("%s%5i \n", padding, ((int8_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_INT16: \
		printf("%s%7i \n", padding, ((int16_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_INT32: \
		printf("%s%12i \n", padding, ((int32_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_INT64: \
		printf("%s%22li \n", padding, ((int64_t *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_FLOAT32: \
		printf("%s%14.5e \n", padding, ((float *)tmp_buf)[_off[0]++]); \
		break; \
	case WEKUA_DTYPE_FLOAT64: \
		printf("%s%14.5e \n", padding, ((double *)tmp_buf)[_off[0]++]); \
		break; \
} \

static const char formats[][6] = {
	"5u",
	"7u",
	"12u",
	"22lu",
	"5i",
	"7i",
	"12i",
	"22li",
	"14.5e",
	"14.5e"
};

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
	cl_command_queue cmd, cl_mem buf, uint64_t offset, uint64_t *_off, void *tmp_buf,
	uint64_t y, uint64_t limit_r, uint64_t c, uint64_t limit_c, uint32_t dtype_length,
	cl_event *events, uint32_t *nevents
){
	for (; y<limit_r; y++){
		uint64_t len = dtype_length*limit_c;
		int ret = clEnqueueReadBuffer(cmd, buf, CL_FALSE, offset + *_off, len, tmp_buf + *_off, 0, NULL, &events[nevents[0]++]);
		assert(ret == CL_SUCCESS);

		_off[0] += len;
		if (c > 4){
			len = MAX((c - 4), 4) * dtype_length;
			ret = clEnqueueReadBuffer(cmd, buf, CL_FALSE, offset + *_off, len, tmp_buf + *_off, 0, NULL, &events[nevents[0]++]);
			assert(ret == CL_SUCCESS);

			_off[0] += len;
		}
	}
}

static void print_real_elements(
	char *padding, char *fmt, uint64_t y, uint64_t limit_r, uint64_t c, uint64_t limit_c,
	void *tmp_buf, uint64_t *_off, uint8_t dtype
){
	for (; y<limit_r; y++){
		uint64_t x = 0;
		for (; x<limit_c; x++){
			PRINT_REAL_VALUES
		}

		if (c > 4) {
			if (c > 8) printf(" ... ");

			x = MAX((c - 4), 4);
			for (; x<c; x++){
				PRINT_REAL_VALUES
			}
		}
	}
}

static void print_matrix(
	wtensor tensor, char *padding, uint64_t ndim, uint64_t *shape, uint64_t *strides,
	uint64_t offset, void *tmp_buf
){
	uint64_t r = shape[0];
	uint64_t c = shape[1];

	cl_mem buf = tensor->buffer;
	wekuaContext ctx = tensor->ctx;
	uint8_t dtype = tensor->dtype;
	uint8_t com = tensor->com;

	uint32_t dtype_length = ctx->dtype_length[dtype];
	const char *format = formats[dtype];
	char *fmt;

	if (com) asprintf(&fmt, "%%%s%%+%sj", format, format);
	else asprintf(&fmt, "%%%s", format);

	cl_event events[16];
	uint32_t nevents = 0;
	cl_command_queue cmd = ctx->command_queue;

	uint64_t limit_r = MIN(r, 4);
	uint64_t limit_c = MIN(c, 4);
	uint64_t _off = 0;
	uint64_t y = 0;

	get_data_from_device(cmd, buf, offset, &_off, tmp_buf, y, limit_r, c, limit_c, dtype_length, events, &nevents);
	if (r > 4){
		y = MAX((r - 4), 4);
		get_data_from_device(cmd, buf, offset, &_off, tmp_buf, y, r, c, limit_c, dtype_length, events, &nevents);
	}
}

static void print_tensor(
	wtensor tensor, char *padding, uint64_t ndim, uint64_t *shape, uint64_t *strides,
	uint64_t offset, void *tmp_buf
){
	printf("%s[", padding);

	padding -= 2;

	if (ndim > 2){
		uint64_t len = shape[0], x = 0;
		uint64_t y = MIN(len, 3);
		uint64_t stride = strides[0];
		for (; x < y; x++){
			print_tensor(tensor, padding, ndim - 1, shape + 1, strides + 1, offset + x * stride, tmp_buf);
		}

		if (len > 6) printf("%s...\n", padding);
		if (y == 3) y = MAX((len - 3), x);

		for (; x<len; x++){
			print_tensor(tensor, padding, ndim - 1, shape + 1, strides + 1, offset + x * stride, tmp_buf);
		}
		printf("\n");
	}else{
		print_matrix(tensor, padding, ndim, shape, strides, offset, tmp_buf);
	}

	printf("%s]", padding + 2);
}

void wekuaTensorPrint(wtensor tensor, uint32_t nw, cl_event *events){
	if (!tensor) return;





}
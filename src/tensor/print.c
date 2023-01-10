#include "tensor.h"
#include <sys/param.h>

static const char formats[][6] = {
	"5i",
	"5u",
	"7i",
	"7u",
	"12i",
	"12u",
	"22li",
	"22lu",
	"14.5e",
	"14.5e"
};

static const char dtype_text[][15] = {
	"int8", "complex_int8",
	"int16", "complex_int16",
	"int32", "complex_int32",
	"int64", "complex_int64",
	"uint8", "complex_uint8",
	"uint16", "complex_uint16",
	"uint32", "complex_uint32",
	"uint64", "complex_uint64",
	"float32", "complex_float32",
	"float64", "complex_float64"
};

void wekuaTensorPrint(wtensor tensor, ...){
	if (!tensor) return;

	wekuaContext ctx = tensor->ctx;
	uint64_t ndim = tensor->ndim;
	uint64_t offset = 0;
	uint64_t *shape = tensor->shape;
	uint64_t *vl_shape = tensor->vl_shape;
	uint64_t dl = ctx->dtype_length[tensor->dtype];

	uint8_t dtype = tensor->dtype;

	va_list ap;
	va_start(ap, tensor);

	if (ndim > 2){
		uint64_t col = tensor->nelements;
		for (uint64_t i = 0; i<(ndim-2); i++){
			col /= vl_shape[i];
			offset += va_arg(ap, uint64_t)*col;
		}
	}

	uint64_t r = shape[ndim-2];
	uint64_t c = shape[ndim-1];
	uint64_t _r = ((r > 8) ? 8 : r);
	uint64_t _c = ((c > 8) ? 8 : c);

	uint8_t v_split = (c > 8) ? 1 : 0;
	uint8_t h_split = (r > 8) ? 1 : 0;

	void *tmp_buffer = calloc(_r * _c, dl);
	if (!tmp_buffer) return;

	uint32_t nw = va_arg(ap, uint32_t);
	cl_event *be = va_arg(ap, cl_event*);
	clWaitForEvents(nw, be);

	


}
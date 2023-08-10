#include "tensor.h"

int wekuaTensorNull(wtensor tensor, uint32_t nw, cl_event *be, cl_event *e){
	if (!tensor) return CL_INVALID_MEM_OBJECT;

	uint64_t zero = 0;

	wekuaContext ctx = tensor->ctx;
	int ret = clEnqueueFillBuffer(
		ctx->command_queue,
		tensor->buffer,
		&zero, sizeof(uint64_t),
		0, tensor->size,
		nw, be, e
	);

	return ret;
}

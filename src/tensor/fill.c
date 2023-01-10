#include "tensor.h"
#include <string.h>

static void fill_finish_event(cl_event event, cl_int event_command_exec_status, void *data){
	if (event_command_exec_status == CL_COMPLETE){
		free(data);
	}
}

int wekuaFillTensor(
	wtensor tensor, void *real, void *imag,
	uint32_t nw, cl_event *be, cl_event *e
){
	if (!tensor) return CL_INVALID_MEM_OBJECT;

	// uint8_t pattern[sizeof(uint64_t)*2] = {0};
	

	wekuaContext ctx = tensor->ctx;
	uint8_t dtype = tensor->dtype;
	uint64_t dl = ctx->dtype_length[dtype];
	uint64_t vl = ctx->device.vectors_size[dtype];
	void *pattern = calloc(vl*2, dl);

	if (real) {
		memcpy(pattern, real, dl);
	}
	if (imag) memcpy(pattern + dl, imag, dl);

	int ret =clEnqueueFillBuffer(
		ctx->command_queue,
		tensor->buffer,
		pattern, dl*2,
		0, tensor->size,
		nw, be, e
	);

	if (ret == CL_SUCCESS) {
		ret = clSetEventCallback(*e, CL_COMPLETE, &fill_finish_event, pattern);
		if (ret != CL_SUCCESS){
			clWaitForEvents(1, e);
			free(pattern);
		}
	}
	return ret;
}
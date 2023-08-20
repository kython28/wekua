#include "tensor.h"
#include <string.h>

struct tensor_fill_event_args {
	void *pattern;
	cl_event to_complex_event;
};

static void fill_finish_event(__attribute__((unused)) cl_event event, cl_int event_command_exec_status, void *data){
	
	if (event_command_exec_status == CL_COMPLETE){
		struct tensor_fill_event_args *args = data;
		free(args->pattern);
		if (args->to_complex_event) clReleaseEvent(args->to_complex_event);
		free(data);
	}
}

int wekuaFillTensor(
	wtensor tensor, void *real, void *imag,
	uint32_t nw, cl_event *be, cl_event *e
){
	if (!tensor) return CL_INVALID_MEM_OBJECT;

	wekuaContext ctx = tensor->ctx;
	uint8_t dtype = tensor->dtype;
	uint64_t dl = ctx->dtype_length[dtype];
	uint64_t vl = ctx->device.vectors_size[dtype];
	void *pattern = calloc(2, vl*dl);

	uint64_t fill_nw = nw;
	cl_event *fill_be = be;

	struct tensor_fill_event_args *args = NULL;
	args->pattern = pattern;

	// uint8_t com = tensor->com;
	// if (!com && imag){
	// 	args = calloc(1, sizeof(struct tensor_fill_event_args));
	// 	fill_be = &args->to_complex_event;
	// 	fill_nw = 1;
	// 	if (wekuaTensorEnableComplexNumbers(tensor, nw, be, fill_be) != CL_SUCCESS){
	// 		free(pattern);
	// 		free(args);
	// 		return CL_OUT_OF_RESOURCES;
	// 	}
	// }

	for (uint64_t x=0; x<vl; x++){
		if (real) memcpy(pattern + x*dl, real, dl);
		if (imag) memcpy(pattern + (vl + x)*dl, imag, dl);
	}

	int ret = clEnqueueFillBuffer(
		ctx->command_queue,
		tensor->buffer,
		pattern, dl*2,
		0, tensor->size,
		fill_nw, fill_be, e
	);

	if (ret == CL_SUCCESS) {
		ret = clSetEventCallback(*e, CL_COMPLETE, &fill_finish_event, pattern);
		if (ret != CL_SUCCESS) {
			clWaitForEvents(1, e);			
		}
	}

	if (ret != CL_SUCCESS){
		free(pattern);
		if (args){
			if (args->to_complex_event){
				clWaitForEvents(1, &args->to_complex_event);
				clReleaseEvent(args->to_complex_event);
			}
			free(args);
		}
	}

	return ret;
}

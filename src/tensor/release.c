#include "tensor.h"

#define RELEASE_AN_SIMPLE_ARRAY(obj) \
	if (obj) { \
		free(obj); \
		obj = NULL; \
	} \

int wekuaFreeTensor(wtensor tensor){
	if (!tensor) return 0;

	RELEASE_AN_SIMPLE_ARRAY(tensor->shape);
	RELEASE_AN_SIMPLE_ARRAY(tensor->vl_shape);
	RELEASE_AN_SIMPLE_ARRAY(tensor->strides);
	RELEASE_AN_SIMPLE_ARRAY(tensor->vl_strides);
	RELEASE_AN_SIMPLE_ARRAY(tensor->work_items);

	int ret = clReleaseMemObject(tensor->buffer);
	if (ret == CL_SUCCESS) free(tensor);
	return ret;
}

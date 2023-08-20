#include "tensor.h"
#include <string.h>

#define CREATE_AN_SIMPLE_ARRAY(name, type) \
	type *name = (type *)calloc(ndim, sizeof(type)); \
	if (!name) goto wekuaTensorEmpty_fail; \
	tensor->name = name; \

wtensor wekuaTensorEmpty(wekuaContext ctx, uint32_t ndim, uint64_t *shape_, uint8_t com, uint8_t dtype){
	if (!ctx || dtype > 9 || ndim == 0) return NULL;

	wtensor tensor = calloc(1, sizeof(struct _w_tensor));
	if (!tensor) return NULL;

    tensor->dtype = dtype;
	tensor->ctx = ctx;
    tensor->ndim = ndim;
	tensor->com = com;

	CREATE_AN_SIMPLE_ARRAY(shape, uint64_t)
	CREATE_AN_SIMPLE_ARRAY(vl_shape, uint64_t)
	CREATE_AN_SIMPLE_ARRAY(strides, uint64_t)
	CREATE_AN_SIMPLE_ARRAY(vl_strides, uint64_t)

	memcpy(shape, shape_, ndim * sizeof(uint64_t));
	memcpy(vl_shape, shape, ndim*sizeof(uint64_t));

	uint64_t nelements = 1;
	for (uint64_t i = 0; i < ndim; i++){
		nelements *= shape[i];
	}

	uint64_t vl = (uint64_t) ctx->device.vectors_size[dtype];
	uint64_t dl = (uint64_t) ctx->dtype_length[dtype];
	uint64_t max = ctx->device.max_work_group_size;
	uint64_t columns = shape[ndim-1];

	uint64_t vl_nelements = nelements/columns;
	if (columns%vl) columns += vl - columns%vl;
	columns /= vl;

	vl_shape[ndim-1] = columns;
	vl_nelements *= columns;
	if (com) vl_nelements *= 2;

	nelements = vl_nelements * vl;
	tensor->nelements = nelements;
	tensor->vl_nelements = vl_nelements;

	uint64_t size = nelements * dl;

	int ret = CL_SUCCESS;
	cl_mem buffer = clCreateBuffer(ctx->ctx, ctx->mem_flags, size, NULL, &ret);
	if (ret != CL_SUCCESS) goto wekuaTensorEmpty_fail;

	tensor->buffer = buffer;
	tensor->size = size;

	uint64_t *work_items = (uint64_t*) calloc(ndim*4 + 2, sizeof(uint64_t));
	if (!work_items) goto wekuaTensorEmpty_fail;

	get_local_work_items(vl_shape, work_items, ndim, max);
	get_local_work_items(shape, work_items + ndim*2, ndim, max);
	get_local_work_items(&vl_nelements, work_items + 4*ndim, 1, max);
	get_local_work_items(&nelements, work_items + 4*ndim, 1, max);
	for (uint64_t i = 0; i < ndim; i++) {
		get_local_work_items(vl_shape + i, work_items + ndim + i, 1, max);
		get_local_work_items(shape + i, work_items + 3*ndim + i, 1, max);
	}
	tensor->work_items = work_items;

	for (uint64_t x=0; x < (ndim - 1); x++){
		nelements /= shape[x];
		strides[x] = nelements;
	}
	strides[ndim - 1] = 1;

	goto wekuaTensorEmpty_success;
	wekuaTensorEmpty_fail:
	wekuaFreeTensor(tensor);
	tensor = NULL;

	wekuaTensorEmpty_success:
	return tensor;
}

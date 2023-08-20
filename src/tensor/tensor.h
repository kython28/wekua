#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "../../headers/tensor.h"
#include "../../headers/utils.h"

typedef struct _w_tensor_common_kernels {
	void (*release_cl_kernels)(struct _w_kernel*);

	cl_program real_program;
	cl_program imag_program;

	cl_kernel real_kernel;
	cl_kernel imag_kernel;
} *w_tensor_common_kernels_t;

#endif

#ifndef WEKUA_ERROR_H
#define WEKUA_ERROR_H

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _w_error {
	wmatrix err; // Error derivate
	void *o_err; // Other errors :v
	uint64_t no_err;
} *werror;

int wekuaMSE(wmatrix output, wmatrix output_wanted, void *error, void *errori, werror *err, uint32_t nw, cl_event *be) __attribute__ ((warn_unused_result)); // Mean Square Error

#ifdef __cplusplus
}
#endif
#endif

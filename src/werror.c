#include "wekua.h"

void wekuaErrorFree(werror error, uint32_t nw, cl_event *be){
	wekuaFreeMatrix(error->err, nw, be);

	wmatrix *er = error->o_err;
	uint64_t n_error = error->no_err;

	for (uint64_t x=0; x<n_error; x++){
		wekuaFreeMatrix(er[x], 0, NULL);
	}

	free(error);
}

int wekuaMSE(wmatrix output, wmatrix output_wanted, void *error, void *errori, werror *err, uint32_t nw, cl_event *be){
	if (output == NULL || output_wanted == NULL) return CL_INVALID_MEM_OBJECT;

	cl_event e[3];
	int ret;

	wmatrix tmp = wekuaMatrixCopy(output_wanted, nw, be, e);
	if (tmp == NULL) return CL_MEM_OBJECT_ALLOCATION_FAILURE;

	ret = wekuaMatrixSub(tmp, output, 1, e, &e[1]);
	if (ret != CL_SUCCESS) goto w_error_mse_fail;

	ret = wekuaMatrixDot(tmp, tmp, 1, &e[1], &e[2]);
	if (ret != CL_SUCCESS) goto w_error_mse_fail;

	


	w_error_mse_fail:
	wekuaFreeMatrix(tmp, 0, NULL);

	return ret;
}
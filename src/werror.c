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

	cl_event e[5];
	uint32_t env = 0;
	int ret;

	wmatrix tmp = wekuaMatrixCopy(output_wanted, nw, be, e);
	if (tmp == NULL) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
	env++;

	ret = wekuaMatrixSub(tmp, output, 1, e, &e[1]);
	if (ret != CL_SUCCESS) goto w_error_mse_fail;
	env++;

	if (err != NULL){
		void *two;

		err[0] = (werror) calloc(1, sizeof(struct _w_error));
		err[0]->err = wekuaMatrixCopy(tmp, 1, &e[env-1], &e[env]);
		if (err[0]->err == NULL){
			wekuaErrorFree(err[0], 0, NULL);
			goto w_error_mse_fail;
		}

		if (tmp->dtype == WEKUA_DTYPE_FLOAT){
			float twof = -2.0;
			two = &twof;
		}else{
			double twod = -2.0;
			two = &twod;
		}
		ret = wekuaBlasScalar(err[0]->err, two, NULL, 1, &e[env-1], &e[env]);
		if (ret != CL_SUCCESS){
			wekuaErrorFree(err[0], 1, &e[env-1]);
			goto w_error_mse_fail;
		}
		env++;
	}

	ret = wekuaMatrixDot(tmp, tmp, 1, &e[env-1], &e[env]);
	if (ret != CL_SUCCESS) goto w_error_mse_fail;
	env++;

	ret = wekuaMatrixMean(tmp, error, errori, 1, &e[env-1]);
	if (ret != CL_SUCCESS) goto w_error_mse_fail;

	w_error_mse_fail:
	if (env > 0) clWaitForEvents(env, e);
	for (uint32_t i=0; i<env; i++) clReleaseEvent(e[i]);

	wekuaFreeMatrix(tmp, 0, NULL);

	return ret;
}
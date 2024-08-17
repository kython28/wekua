#include "../headers/acti.h"

int runWekuaActi(wacti acti, wmatrix input, uint32_t nw, cl_event *be){
	if (acti == NULL || input == NULL) return CL_INVALID_MEM_OBJECT;

	return acti->run_acti(acti->data, input, nw, be);
}

void wekuaFreeActi(wacti acti, uint32_t nw, cl_event *be){
	if (acti == NULL) return;

	acti->free_func(acti, nw, be);
}

wmatrix wekuaActiGetDev(wacti acti, wmatrix output){
	if (acti == NULL || output == NULL) return NULL;
	return acti->get_dev(acti->data, output);
}


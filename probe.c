#include "src/wekua.h"
#include <sys/time.h>
#include <time.h>


int main(){
	wekuaContext ctx = createSomeWekuaContext(CL_DEVICE_TYPE_CPU);
	if (ctx == NULL) return 1;

	double buf[] = {
		1.0, 1.0,
		0.0, 1.0,
		1.0, 0.0,
		0.0, 0.0
	};

	wmatrix a = wekuaMatrixFromBuffer(ctx, 4, 2, buf, NULL, WEKUA_DTYPE_DOUBLE);

	wneuron lik = wekuaLinear(ctx, 2, 1, 1, 1, wekuaActiTanh(), WEKUA_DTYPE_DOUBLE);

	struct timeval start, end;
	cl_event e;

	wmatrix b = lik->run(lik, a, NULL, 0, NULL);
	wekuaMatrixPrint(b, 0, 0);
	gettimeofday(&start, 0);

	// wekuaBlasScalar(a, &one, NULL, 0, NULL, &e);
	// clWaitForEvents(1, &e);
	// clReleaseEvent(e);

	b = lik->run(lik, a, NULL, 0, NULL);

	gettimeofday(&end, 0);
	
	wekuaMatrixPrint(a, 0, 0);
	wekuaMatrixPrint(b, 0, 0);

	printf("Took: %ld us\n", (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec));

	wekuaFreeMatrix(a, 0, 0);
	freeWekuaContext(ctx);
	// freeWekuaContext(ctx2);
	return 0;
}
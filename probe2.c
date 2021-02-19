#include <wekua.h>

int main(int argc, char const *argv[])
{
	wekuaContext ctx = createSomeWekuaContext(CL_DEVICE_TYPE_CPU);
	if (ctx == NULL) return 1;

	double alpha = 1.0;

	cl_event e;
	wmatrix a = wekuaMatrixRandn(ctx, 200, 200, 1);
	wmatrix b = wekuaMatrixInv(a, 0, NULL);
	wmatrix c = wekuaAllocMatrix(ctx, 200, 200, WEKUA_DTYPE_DOUBLE);

	// clWaitForEvents(1, &e);
	// clReleaseEvent(e);

	wekuaBlasGemm(&alpha, NULL, 0, a, 0, b, NULL, NULL, c, 0, NULL);

	// clWaitForEvents(1, &e);
	// clReleaseEvent(e);

	wekuaMatrixPrint(a, 0, NULL);
	wekuaMatrixPrint(b, 0, NULL);
	wekuaMatrixPrint(c, 0, NULL);

	// clReleaseEvent(e);

	wekuaFreeMatrix(a, 0, NULL);
	wekuaFreeMatrix(b, 0, NULL);
	wekuaFreeMatrix(c, 0, NULL);

	freeWekuaContext(ctx);

	return 0;
}
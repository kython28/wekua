#include "src/wekua.h"
#include <sys/time.h>
#include <time.h>

#define N 1000

int main(){
	wekuaContext ctx = createSomeWekuaContext(CL_DEVICE_TYPE_CPU);
	if (ctx == NULL) return 1;

	double one = 1.0;

	struct timeval start, end;
	cl_event e;

	float inputs[8] = {
		1.0, 1.0,
		1.0, 0.0,
		0.0, 1.0,
		0.0, 0.0
	};

	float outputs[4] = {
		1.0,
		0.0,
		0.0,
		0.0
	};

	float error;

	wmatrix input = wekuaMatrixFromBuffer(ctx, 4, 2, inputs, NULL, WEKUA_DTYPE_FLOAT);
	wmatrix output_wanted = wekuaMatrixFromBuffer(ctx, 4, 1, outputs, NULL, WEKUA_DTYPE_FLOAT);

	wacti acti = wekuaActiSigmoid();

	wcache *cache;
	wnetwork net = wekuaNeuronNetwork(1, WEKUA_DTYPE_FLOAT);
	net->neurons[0] = wekuaLinear(ctx, 2, 1, 1, 1, acti, WEKUA_DTYPE_FLOAT); 
	
	gettimeofday(&start, 0);

	wmatrix output = runWekuaNetwork(net, input, &cache);

	gettimeofday(&end, 0);

	werror error_dev, *error_dev2;

	for (uint64_t x=0; x<cache[0]->ndata; x++){
		wekuaMatrixPrint(((wmatrix*)cache[0]->data)[x], 0, NULL);
	}

	wekuaMSE(output, output_wanted, &error, NULL, &error_dev, 0, NULL);

	wekuaNetworkBackward(net, error_dev, cache, &error_dev2);

	wekuaMatrixPrint(error_dev2[0]->err, 0, NULL);
	wekuaMatrixPrint(output_wanted, 0, NULL);
	wekuaMatrixPrint(output, 0, NULL);

	printf("Took: %ld us\n", (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec));

	printf("%f\n", error);
	wekuaFreeMatrix(input, 0, NULL);
	wekuaFreeMatrix(output, 0, NULL);

	freeWekuaContext(ctx);
	// freeWekuaContext(ctx2);
	return 0;
}
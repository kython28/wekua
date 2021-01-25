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

	double inputs[8] = {
		1.0, 1.0,
		1.0, 0.0,
		0.0, 1.0,
		0.0, 0.0
	};

	double outputs[4] = {
		0.0,
		1.0,
		1.0,
		0.0
	};

	double error = 1.0;
	double alpha = 0.5;

	uint32_t total = 0, t = 0;

	wmatrix input = wekuaMatrixFromBuffer(ctx, 4, 2, inputs, NULL, WEKUA_DTYPE_DOUBLE);
	wmatrix output_wanted = wekuaMatrixFromBuffer(ctx, 4, 1, outputs, NULL, WEKUA_DTYPE_DOUBLE);

	wacti acti = wekuaActiSigmoid();

	wcache *cache;
	wnetwork net = wekuaNeuronNetwork(2, WEKUA_DTYPE_DOUBLE);
	net->neurons[0] = wekuaLinear(ctx, 2, 20, 1, 1, acti, WEKUA_DTYPE_DOUBLE);
	net->neurons[1] = wekuaLinear(ctx, 20, 1, 1, 1, acti, WEKUA_DTYPE_DOUBLE);
	wmatrix output;

	werror *error_dev;
	error_dev = (werror*) calloc(2, sizeof(werror));

	woptim optim = wekuaOptimGD(ctx, &alpha, NULL, WEKUA_DTYPE_DOUBLE);
	int ret;

	for (uint32_t i = 0; i < 20000; i++){
		gettimeofday(&start, 0);

		output = runWekuaNetwork(net, input, &cache);
		ret = wekuaMSE(output, output_wanted, &error, NULL, error_dev, 0, NULL);
		ret = wekuaNetworkBackward(net, error_dev, cache, NULL);
		ret = wekuaOptimStep(optim, net, error_dev, cache);

		wekuaFreeNetCache(net, cache);
		wekuaFreeNetError(net, error_dev);
		
		gettimeofday(&end, 0);

		t += (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
		total++;

		printf("%d) Took: %ld us -> %e\n", i, t/total, error);
	}

	
	output = runWekuaNetwork(net, input, &cache);
	wekuaMatrixPrint(output_wanted, 0, NULL);
	wekuaMatrixPrint(output, 0, NULL);

	// printf("Took: %ld us\n", (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec));

	printf("%f\n", error);
	wekuaFreeMatrix(input, 0, NULL);
	wekuaFreeMatrix(output, 0, NULL);

	freeWekuaContext(ctx);
	// freeWekuaContext(ctx2);
	return 0;
}
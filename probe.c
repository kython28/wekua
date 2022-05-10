#include <wekua/network.h>
#include <wekua/optim.h>
#include <stdio.h>

#include <time.h>

int main(){
	wekuaContext ctx = createSomeWekuaContext(CL_DEVICE_TYPE_GPU, 1);
	if (ctx == NULL) return 1;

	double alpha = 2.0, error = 1.0f;
	struct timespec tstart={0,0}, tend={0,0};

	double input_buf[] = {
		1.0f, 1.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,
		0.0f, 0.0f
	};

	double output_buf[] = {
		1.0f, 0.0f, 0.0f, 0.0f
	};

	wmatrix input = wekuaMatrixFromBuffer(ctx, 4, 2, input_buf, NULL, WEKUA_DTYPE_DOUBLE);
	wmatrix output_wanted = wekuaMatrixFromBuffer(ctx, 4, 1, output_buf, NULL, WEKUA_DTYPE_DOUBLE);

	wekuaMatrixPrint(input, 0, NULL);
	wekuaMatrixPrint(output_wanted, 0, NULL);

	wacti sigmoid = wekuaActiSigmoid();

	wnetwork net = wekuaNeuronNetwork(1, WEKUA_DTYPE_DOUBLE);
	net->neurons[0] = wekuaLinear(ctx, 2, 1, 1, 1, sigmoid, WEKUA_DTYPE_DOUBLE);

	wcache *cache;
	werror *err = calloc(1, sizeof(werror));

	woptim optim = wekuaOptimAdaGrad(ctx, net, &alpha, NULL, WEKUA_DTYPE_DOUBLE);

	wmatrix output;

	for (uint32_t n=0; n<10000;){
		clock_gettime(CLOCK_MONOTONIC, &tstart);
		for (uint32_t x=0; x<1000; x++){
			output = runWekuaNetwork(net, input, &cache);

			wekuaCrossEntropy(output, output_wanted, &error, NULL, err, 0, NULL);
			wekuaNetworkBackward(net, err, cache, NULL);
			wekuaOptimStep(optim, err, cache);

			wekuaFreeNetError(net, err);
			wekuaFreeNetCache(net, cache);
		}
		clock_gettime(CLOCK_MONOTONIC, &tend);
		n += 1000;
		printf("Epoch: %d - Error: %.8f - Took %.6f\n", n, error, (((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec))/1000);
	}

	output = runWekuaNetwork(net, input, 0);
	wekuaMatrixPrint(output, 0, 0);

	wekuaFreeMatrix(output, 0, 0);

	free(err);
	wekuaFreeActi(sigmoid, 0, 0);
	wekuaFreeNetwork(net, 0, 0);

	wekuaFreeMatrix(output_wanted, 0, NULL);
	wekuaFreeMatrix(input, 0, NULL);

	freeWekuaContext(ctx);
	return 0;
}

#include <wekua.h>

int main(){
	wekuaContext ctx = createSomeWekuaContext(CL_DEVICE_TYPE_CPU, 1);

	float alpha = 2.0, error = 1.0f;

	float input_buf[] = {
		1.0f, 1.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,
		0.0f, 0.0f
	};

	float output_buf[] = {
		1.0f, 0.0f, 0.0f, 0.0f
	};

	wmatrix input = wekuaMatrixFromBuffer(ctx, 4, 2, input_buf, NULL, WEKUA_DTYPE_FLOAT);
	wmatrix output_wanted = wekuaMatrixFromBuffer(ctx, 4, 1, output_buf, NULL, WEKUA_DTYPE_FLOAT);

	wacti sigmoid = wekuaActiSigmoid();

	wnetwork net = wekuaNeuronNetwork(1, WEKUA_DTYPE_FLOAT);
	net->neurons[0] = wekuaLinear(ctx, 2, 1, 1, 1, sigmoid, WEKUA_DTYPE_FLOAT);

	loadWekuaNetwork("we", net, ctx);

	wcache *cache;
	werror *err = calloc(1, sizeof(werror));

	woptim optim = wekuaOptimGD(ctx, net, &alpha, NULL, WEKUA_DTYPE_FLOAT);

	wmatrix output;

	for (uint32_t n=0; n<10000;){
		for (uint32_t x=0; x<1000; x++){
			output = runWekuaNetwork(net, input, &cache);

			wekuaMSE(output, output_wanted, &error, NULL, err, 0, NULL);
			wekuaNetworkBackward(net, err, cache, NULL);
			wekuaOptimStep(optim, err, cache);

			wekuaFreeNetError(net, err);
			wekuaFreeNetCache(net, cache);
		}
		n += 1000;
		printf("Epoch: %d - Error: %.8f\n", n, error);
	}

	output = runWekuaNetwork(net, input, 0);
	wekuaMatrixPrint(output, 0, 0);

	wekuaFreeMatrix(output, 0, 0);

	saveWekuaNetwork("we", net);

	free(err);
	wekuaFreeActi(sigmoid, 0, 0);
	wekuaFreeNetwork(net, 0, 0);

	wekuaFreeMatrix(output_wanted, 0, NULL);
	wekuaFreeMatrix(input, 0, NULL);

	freeWekuaContext(ctx);
	return 0;
}
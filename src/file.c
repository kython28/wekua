#include "wekua.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

struct header {
	uint64_t r; // row
	uint64_t c; // header
	uint64_t size;
	uint8_t dtype;
	uint8_t com; // Does the matrix use complex elements?
};

int open_a_file(const char *name, uint8_t w){
	int fd;
	if (w){
		fd = open(name, O_WRONLY|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
		if (fd < 0){
			fd = open(name, O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
			if (fd < 0) return -1;
		}
	}else{
		fd = open(name, O_RDWR);
		if (fd < 0) return -1;
	}
	return fd;
}

uint8_t saveMatrix(int fd, wmatrix a){
	uint8_t ret = CL_SUCCESS;

	uint64_t size = 0, r, c;
	uint8_t dtype = a->dtype, com = a->com;
	uint32_t dl;

	void *real, *imag;

	real = a->raw_real;
	imag = a->raw_imag;

	wekuaContext ctx = a->ctx;
	dl = ctx->dtype_length[dtype];

	struct header file_h;

	r = a->shape[0];
	c = a->col;

	size = a->shape[1];

	file_h.r = r;
	file_h.c = size;

	size *= dl;

	file_h.size = size*r;
	file_h.dtype = dtype;
	file_h.com = com;

	if (write(fd, &file_h, sizeof(struct header)) != sizeof(struct header)){
		ret = 1;
		goto save_finish;
	}

	c *= dl;
	r *= c;

	for (uint64_t i=0; i<r; i+=c){
		if (write(fd, real + i, size) != size){
			ret = 1;
			break;
		}
	}

	if (com){
		for (uint64_t i=0; i<r; i+=c){
			if (write(fd, imag + i, size) != size){
				ret = 1;
				break;
			}
		}
	}

	save_finish:
	return ret;
}

wmatrix loadMatrix(int fd, wekuaContext ctx){
	wmatrix a = NULL;
	struct header file_h;

	if (read(fd, &file_h, sizeof(struct header)) != sizeof(struct header)) return NULL;

	void *real, *imag;
	uint64_t c1 = file_h.c, c2, r = file_h.r;
	uint32_t dl = ctx->dtype_length[file_h.dtype];

	if (file_h.com){
		a = wekuaAllocComplexMatrix(ctx, file_h.r, file_h.c, file_h.dtype);
	}else{
		a = wekuaAllocMatrix(ctx, file_h.r, file_h.c, file_h.dtype);
	}

	c2 = a->col;
	real = a->raw_real;
	imag = a->raw_imag;

	c1 *= dl;
	c2 *= dl;
	r *= c2;

	for (uint64_t i=0; i<r; i += c2){
		if (read(fd, real+i, c1) != c1){
			wekuaFreeMatrix(a, 0, NULL);
			a = NULL;
			goto load_finish;
		}
	}

	if (file_h.com){
		for (uint64_t i=0; i<r; i += c2){
			if (read(fd, imag+i, c1) != c1){
				wekuaFreeMatrix(a, 0, NULL);
				a = NULL;
				break;
			}
		}
	}

	load_finish:
	if (real != NULL) munmap(real, file_h.size);
	if (imag != NULL) munmap(imag, file_h.size);
	return a;
}


uint8_t saveNeuron(int fd, wneuron neuron){
	uint64_t layer = neuron->layer;
	for (uint64_t x=0; x<layer; x++){
		if (saveMatrix(fd, neuron->weight[x])) break;
	}
	if (neuron->bias){
		for (uint64_t x=0; x<layer; x++){
			if (saveMatrix(fd, neuron->bias[x])) break;
		}
	}
	return 0;
}

void loadNeuron(int fd, wekuaContext ctx, wneuron neuron){
	wmatrix *w, *b;
	uint64_t layer = neuron->layer;

	w = neuron->weight;
	b = neuron->bias;
	for (uint64_t x=0; x<layer; x++){
		w[x] = loadMatrix(fd, ctx);
	}
	if (b){
		for (uint64_t x=0; x<layer; x++){
			b[x] = loadMatrix(fd, ctx);
		}
	}
}

uint8_t saveWekuaMatrix(const char *name, wmatrix a){
	if (name == NULL || a == NULL) return 1;

	int fd;
	uint8_t ret;
	fd = open_a_file(name, 1);
	if (fd < 0) return 1;

	ret = saveMatrix(fd, a);

	close(fd);
	return ret;
}

wmatrix loadWekuaMatrix(const char *name, wekuaContext ctx){
	if (name == NULL || ctx == NULL) return NULL;
	wmatrix a;

	int fd = open_a_file(name, 0);
	if (fd < 0) return NULL;

	a = loadMatrix(fd, ctx);

	return a;
}

uint8_t saveWekuaNeuron(const char *name, wneuron neuron){
	if (name == NULL || neuron == NULL) return 1;
	int fd;
	uint8_t ret;
	uint64_t layer = neuron->layer;

	fd = open_a_file(name, 1);
	if (fd < 0) return 1;

	ret = saveNeuron(fd, neuron);
	close(fd);
	return ret;
}

uint8_t loadWekuaNeuron(const char *name, wneuron neuron){
	if (name == NULL || neuron == NULL) return 1;
	int fd = open_a_file(name, 0);
	if (fd < 0) return 1;

	wmatrix *w, *b;
	wekuaContext ctx;
	uint64_t layer = neuron->layer;

	w = neuron->weight;
	b = neuron->bias;

	ctx = w[0]->ctx;

	for (uint64_t x=0; x<layer; x++) wekuaFreeMatrix(w[x], 0, NULL);
	if (b){
		for (uint64_t x=0; x<layer; x++) wekuaFreeMatrix(b[x], 0, NULL);
	}

	loadNeuron(fd, ctx, neuron);

	close(fd);
	return 0;
}

uint8_t saveWekuaNetwork(const char *name, wnetwork net){
	if (name == NULL || net == NULL) return 1;

	uint8_t ret = 0;
	int fd = open_a_file(name, 1);
	if (fd < 0) return 1;

	uint32_t nneur = net->nneur;
	wneuron *neurons = net->neurons;
	for (uint32_t x=0; x<nneur; x++){
		if (saveNeuron(fd, neurons[x])){
			ret = 1;
			break;
		}
	}

	close(fd);
	return ret;
}

uint8_t loadWekuaNetwork(const char *name, wnetwork net, wekuaContext ctx){
	if (name == NULL || net == NULL || ctx == NULL) return 1;

	uint8_t ret = 0;
	int fd = open_a_file(name, 0);
	if (fd < 0) return 1;

	uint32_t nneur = net->nneur;
	wneuron *neurons = net->neurons;
	for (uint32_t x=0; x<nneur; x++){
		wmatrix *w, *b;
		wneuron neuron = neurons[x];
		uint64_t layer = neuron->layer;
		w = neuron->weight;
		b = neuron->bias;
		
		for (uint64_t x=0; x<layer; x++) wekuaFreeMatrix(w[x], 0, NULL);
		if (b){
			for (uint64_t x=0; x<layer; x++) wekuaFreeMatrix(b[x], 0, NULL);
		}

		loadNeuron(fd, ctx, neuron);
	}

	close(fd);
	return ret;
}
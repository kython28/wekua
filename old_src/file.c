#include "../headers/matrix.h"
#include "../headers/network.h"
#include "../headers/neuron.h"

#include <unistd.h>
#include <sys/mman.h>

#define _GNU_SOURCE
#include <fcntl.h>

struct header {
	uint64_t r; // row
	uint64_t c; // header
	uint64_t size;
	uint8_t dtype;
	uint8_t com; // Does the matrix use complex elements?
};

int open_a_file(const char *name, uint8_t create){
	int fd;
	if (create){
		fd = open(name, O_RDWR|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
		if (fd < 0){
			fd = open(name, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
		}
	}else{
		fd = open(name, O_RDWR);
	}

	return fd;
}

uint8_t saveMatrix(int fd, wmatrix a){
	uint8_t ret = CL_SUCCESS;

	uint64_t size = 0, r;
	uint8_t dtype = a->dtype, com = a->com;
	uint32_t dl;

	wekuaContext ctx = a->ctx;
	dl = ctx->dtype_length[dtype];

	struct header file_h;

	r = a->shape[0];

	size = a->shape[1];

	file_h.r = r;
	file_h.c = size;

	size *= dl*r;

	file_h.size = size;
	file_h.dtype = dtype;
	file_h.com = com;

	if (write(fd, &file_h, sizeof(struct header)) != sizeof(struct header)){
		ret = 1;
		goto save_finish;
	}

	if (com) size *= 2;

	ssize_t lseek_ret = lseek(fd, (ssize_t)size, SEEK_CUR);
	if (lseek_ret < 0) return 1;

	uint64_t current = (uint64_t) lseek_ret;
	posix_fallocate(fd, (ssize_t)current, (ssize_t)size);

	void *addr = mmap(NULL, current+size, PROT_WRITE, MAP_SHARED, fd, 0);

	if (com) {
		 if (wekuaMatrixWritetoBuffer(a, addr + current, addr + current + size/2) != CL_SUCCESS) {
			 return 1;
		 }
	}
	else {
		if (wekuaMatrixWritetoBuffer(a, addr + current, NULL) != CL_SUCCESS) {
			return 1;
		}
	}

	munmap(addr, current+size);

	lseek(fd, 0, SEEK_END);

	save_finish:
	return ret;
}

wmatrix loadMatrix(int fd, wekuaContext ctx){
	wmatrix a = NULL;
	struct header file_h;

	if (read(fd, &file_h, sizeof(struct header)) != sizeof(struct header)) return NULL;

	ssize_t lseek_ret = lseek(fd, 0, SEEK_CUR);
	if (lseek_ret < 0) return NULL;

	uint64_t current = (uint64_t)lseek_ret;

	void *addr = NULL;
	uint64_t size = file_h.size;

	if (file_h.com){
		addr = mmap(NULL, current + size*2, PROT_READ, MAP_SHARED, fd, 0);
		a = wekuaMatrixFromBuffer(ctx, file_h.r, file_h.c, addr + current, addr + current + size, file_h.dtype);
		size *= 2;
	}else{
		addr = mmap(NULL, current + size, PROT_READ, MAP_SHARED, fd, 0);
		a = wekuaMatrixFromBuffer(ctx, file_h.r, file_h.c, addr + current, NULL, file_h.dtype);
	}

	munmap(addr, current + size);
	lseek(fd, (ssize_t)(current + size), SEEK_SET);
	return a;
}


uint8_t saveNeuron(int fd, wneuron neuron){
	// -----------------------------------------
	// This is temporary
	uint64_t layer = neuron->layer;
	wmatrix *w, *b;
	w = neuron->weight;
	b = neuron->bias;
	
	for (uint64_t x=0; x<layer; x++){
		if (saveMatrix(fd, w[x])) break;
	}
	if (b){
		for (uint64_t x=0; x<layer; x++){
			if (saveMatrix(fd, b[x])) break;
		}
	}
	// -----------------------------------------
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
		
		for (uint64_t y=0; y<layer; y++) wekuaFreeMatrix(w[y], 0, NULL);
		if (b){
			for (uint64_t y=0; y<layer; y++) wekuaFreeMatrix(b[y], 0, NULL);
		}

		loadNeuron(fd, ctx, neuron);
	}

	close(fd);
	return ret;
}

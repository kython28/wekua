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


uint8_t saveWekuaMatrix(const char *name, wmatrix a){
	if (name == NULL || a == NULL) return CL_INVALID_ARG_VALUE;

	int fd;
	uint8_t ret;
	fd = open(name, O_WRONLY|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
	if (fd < 0){
		fd = open(name, O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
		if (fd < 0) return 1;
	}

	ret = saveMatrix(fd, a);

	close(fd);
	return ret;
}

wmatrix loadWekuaMatrix(const char *name, wekuaContext ctx){
	if (name == NULL || ctx == NULL) return NULL;
	wmatrix a;

	int fd = open(name, O_RDWR);
	if (fd < 0) return NULL;

	a = loadMatrix(fd, ctx);

	return a;
}
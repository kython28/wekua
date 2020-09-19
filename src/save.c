#include "wekua.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

const uint64_t wekua_signature = 0x77656b7561000000;

int openWekuaFile(const char *name, uint8_t type){
	uint64_t size, ws;
	uint8_t typ;
	int fd = open(name, O_RDWR|O_CREAT|O_SYNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
	if (fd < 0){
		return fd;
	}
	size = lseek(fd, 0, SEEK_END);
	lseek(fd, 0, SEEK_SET);
	if (size == 0){ // New file
		if (write(fd, &wekua_signature, 8) != 8){
			goto open_wekua_file_fail;
		}else if (write(fd, &type, 1) != 1){
			goto open_wekua_file_fail;
		}
	}else{ // Check
		if (read(fd, &ws, 8) != 8){
			goto open_wekua_file_fail;
		}else if (ws != wekua_signature){
			goto open_wekua_file_fail;
		}

		if (read(fd, &typ, 1) != 1){
			goto open_wekua_file_fail;
		}else if (typ != type){
			goto open_wekua_file_fail;
		}
	}
	goto open_wekua_file_success;

	open_wekua_file_fail:
	close(fd);
	fd = -1;
	open_wekua_file_success:
	return fd;
}

int writeWekuaFile(int fd, wmatrix *a){
	uint64_t size = a->size;
	if (a->sm){
		size = a->shape[0]*a->shape[1];
	}

	if (write(fd, a->shape, 16) != 16){
		return 1;
	}
	if (write(fd, &size, 8) != 8){
		return 1;
	}
	if (write(fd, &a->com, 1) != 1){
		return 1;
	}
	if (write(fd, a->raw_real, sizeof(double)*size) != sizeof(double)*size){
		return 1;
	}
	if (a->com){
		if (write(fd, a->raw_imag, sizeof(double)*size) != sizeof(double)*size){
			return 1;
		}	
	}

	return 0;
}

wmatrix *readWekuaFile(wekuaContext *ctx, int fd){
	wmatrix *a;
	uint64_t shape[2], size;
	uint8_t com;
	if (read(fd, shape, 16) != 16){
		return NULL;
	}
	if (read(fd, &size, 8) != 8){
		return NULL;
	}
	size *= sizeof(double);

	if (read(fd, &com, 1) != 1){
		return NULL;
	}

	if (com){
		a = wekuaAllocComplexMatrix(ctx, shape[0], shape[1]);
	}else{
		a = wekuaAllocMatrix(ctx, shape[0], shape[1]);
	}

	if (a == NULL){
		return NULL;
	}

	if (read(fd, a->raw_real, size) != size){
		wekuaFreeMatrix(a, 0, NULL);
		return NULL;
	}
	if (com){
		if (read(fd, a->raw_imag, size) != size){
			wekuaFreeMatrix(a, 0, NULL);
			return NULL;
		}
	}
	return a;
}

int saveWekuaMatrix(const char *name, wmatrix *a){
	if (a == NULL || name == NULL){
		return 1;
	}

	int fd = openWekuaFile(name, 0), ret = 0;
	if (fd < 0){
		return 1;
	}
	if (writeWekuaFile(fd, a)){
		ret = 1;
	}
	close(fd);
	return ret;
}

wmatrix *openWekuaMatrix(wekuaContext *ctx, const char *name){
	if (name == NULL){
		return NULL;
	}
	wmatrix *a;
	int fd = openWekuaFile(name, 0);
	if (fd < 0){
		return NULL;
	}
	a = readWekuaFile(ctx, fd);
	close(fd);
	return a;
}

int saveWekuaArch(const char *name, warch *arch){
	if (name == NULL){
		return 1;
	}
	int fd = openWekuaFile(name, 1), ret = 0;
	if (fd < 0){
		return 1;
	}
	for (uint32_t x=0; x < arch->nmodule[2]; x++){
		if (writeWekuaFile(fd, arch->weight[x])){
			ret = 1;
			break;
		}
	}
	close(fd);
	return 0;
}

int openWekuaArch(const char *name, warch *arch){
	if (name == NULL){
		return 1;
	}
	wmatrix *tmp;
	wekuaContext *ctx = arch->weight[0]->ctx;
	int fd = openWekuaFile(name, 1), ret = 0;
	if (fd < 0){
		printf("%d\n", fd);
		return 1;
	}
	wmatrix **wei = arch->weight;
	for (uint32_t x=0; x < arch->nmodule[2]; x++){
		tmp = readWekuaFile(ctx, fd);
		if (tmp == NULL){
			printf("lola\n");
		}
		memcpy(wei[x]->raw_real, tmp->raw_real, tmp->size*sizeof(double));
		if (tmp->com){
			memcpy(wei[x]->raw_imag, tmp->raw_imag, tmp->size*sizeof(double));
		}
		wekuaFreeMatrix(tmp, 0, NULL);
	}
	close(fd);
	return 0;
}
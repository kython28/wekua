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

int saveWekuaMatrix(const char *name, wmatrix *a){
	if (a == NULL || name == NULL){
		return 1;
	}

	int fd = openWekuaFile(name, 0), ret = 1;
	if (fd < 0){
		return 1;
	}
	uint64_t size = a->size;
	if (a->sm){
		size = a->shape[0]*a->shape[1];
	}

	if (write(fd, a->shape, 16) != 16){
		goto save_wekua_matrix_fail;
	}
	if (write(fd, &size, 8) != 8){
		goto save_wekua_matrix_fail;
	}
	if (write(fd, &a->com, 1) != 1){
		goto save_wekua_matrix_fail;
	}
	if (write(fd, a->raw_real, sizeof(double)*size) != sizeof(double)*size){
		goto save_wekua_matrix_fail;
	}
	if (a->com){
		if (write(fd, a->raw_imag, sizeof(double)*size) != sizeof(double)*size){
			goto save_wekua_matrix_fail;
		}	
	}
	ret = 0;
	save_wekua_matrix_fail:
	close(fd);
	return ret;
}

wmatrix *openWekuaMatrix(wekuaContext *ctx, const char *name){
	if (name == NULL){
		return NULL;
	}
	wmatrix *a;
	uint64_t shape[2], size;
	uint8_t com;

	int fd = openWekuaFile(name, 0);
	if (fd < 0){
		goto open_matrix_fail;
	}

	if (read(fd, shape, 16) != 16){
		goto open_matrix_fail;
	}
	if (read(fd, &size, 8) != 8){
		goto open_matrix_fail;
	}
	size *= sizeof(double);

	if (read(fd, &com, 1) != 1){
		goto open_matrix_fail;
	}

	if (com){
		a = wekuaAllocComplexMatrix(ctx, shape[0], shape[1]);
	}else{
		a = wekuaAllocMatrix(ctx, shape[0], shape[1]);
	}

	if (a == NULL){
		goto open_matrix_success;
	}

	if (read(fd, a->raw_real, size) != size){
		wekuaFreeMatrix(a, 0, NULL);
		goto open_matrix_fail;
	}
	if (com){
		if (read(fd, a->raw_imag, size) != size){
			wekuaFreeMatrix(a, 0, NULL);
			goto open_matrix_fail;
		}
	}
	close(fd);

	goto open_matrix_success;
	open_matrix_fail:
	a = NULL;
	open_matrix_success:
	return a;
}
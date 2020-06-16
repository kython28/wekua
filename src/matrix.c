#include "wekua.h"
#include <math.h>

uint64_t getWI(uint64_t a, uint64_t max){
	if (a == 1 || max == 1){
		return 1;
	}else if (a <= max){
		return a;
	}
	uint64_t x;
	for (x=2; max < a/x || a%x != 0; x++);
	return a/x;
}

void getLWI(void *x, void *y, uint32_t si, uint64_t max){
	uint64_t c = (uint64_t) pow(1.0*max, 1.0/si);
	for (uint32_t j=0; j<si; j++){
		if (((uint64_t*)x)[j] < c){
			((uint64_t*)y)[j] = ((uint64_t*)x)[j];
			continue;
		}
		((uint64_t*)y)[j] = c;
		while (((uint64_t*)x)[j]%((uint64_t*)y)[j] != 0){
			((uint64_t*)y)[j]--;
		}
	}
}

void MapBufferMatrix(wmatrix *a){
	if (a->real != NULL && a->raw_real == NULL){
		a->raw_real = clEnqueueMapBuffer(a->ctx->command_queue, a->real, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);	
	}
	if (a->imag != NULL && a->raw_imag == NULL){
		a->raw_imag = clEnqueueMapBuffer(a->ctx->command_queue, a->imag, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*sizeof(double), 0, 0, NULL, NULL);
	}
}

void UnmapBufferMatrix(wmatrix *a){
	if (a->real != NULL && a->raw_real != NULL){
		clEnqueueUnmapMemObject(a->ctx->command_queue, a->real, a->raw_real, 0, NULL, NULL);
		a->raw_real = NULL;
	}
	if (a->imag != NULL && a->raw_imag != NULL){
		clEnqueueUnmapMemObject(a->ctx->command_queue, a->imag, a->raw_imag, 0, NULL, NULL);
		a->raw_imag = NULL;
	}
}

void runKernel(cl_command_queue cmd, cl_kernel kernel, uint32_t ndim, uint64_t *offsi, uint64_t *glosi, uint64_t *losi){
	cl_event event;
	uint32_t ret = clEnqueueNDRangeKernel(cmd, kernel, ndim, offsi, glosi, losi, 0, NULL, &event);
	if (ret != 0){
		printf("Failed to run kernel %i :-(\n", ret);
	}
	ret = clWaitForEvents(1, &event);
	if (ret != 0){
		printf("Failed to run kernel %i :-(\n", ret);
	}
	clReleaseEvent(event);
}

uint8_t createComplexMatrix(wmatrix *a){
	int ret;
	a->imag = clCreateBuffer(a->ctx->ctx, CL_MEM_READ_WRITE, sizeof(double)*a->size, NULL, &ret);
	if (ret != 0){
		printf("Failed to allocate new memory :-(\n");
		return 1;
	}
	MapBufferMatrix(a);
	memset(a->raw_imag, 0, a->size*sizeof(double));
	a->com = 1;
	return 0;
}

void removeComplexMatrix(wmatrix *a){
	clEnqueueUnmapMemObject(a->ctx->command_queue, a->imag, a->raw_imag, 0, NULL, NULL);
	a->raw_imag = NULL;
	clReleaseMemObject(a->imag);
	a->imag = NULL;
	a->com = 0;
}

void wekuaFreeMatrix(wmatrix *a){
	if (a == NULL){
		return;
	}
	UnmapBufferMatrix(a);
	if (a->real != NULL){
		clReleaseMemObject(a->real);
	}
	if (a->imag != NULL){
		clReleaseMemObject(a->imag);
	}
	free(a);
}

void wekuaMatrixRealPrint(wmatrix *a){
	if (a == NULL){
		return;
	}
	for (uint32_t y=0; y<a->r; y++){
		for (uint32_t x=0; x<a->c; x++){
			if (x == 0 && (y < 5 || y >= a->r-4)){
				printf("[");
			}
			if ((x < 4 || x >= a->c-4) && (y < 4 || y >= a->r-4)){
				printf("%14.5e", a->raw_real[y*a->c+x]);
			}else if ((x == 4 && (y < 5 || y >= a->r-4)) || (y == 4 && (x < 4 || x >= a->c-4))){
				printf("%14s", "...");
			}
			if (x == a->c-1 && (y < 5 || y >= a->r-4)){
				printf("]\n");
			}
		}
	}
	printf("\n");
}

void wekuaMatrixComplexPrint(wmatrix *a){
	if (a == NULL){
		return;
	}
	for (uint32_t y=0; y<a->r; y++){
		for (uint32_t x=0; x<a->c; x++){
			if (x == 0 && (y < 3 || y >= a->r-2)){
				printf("[");
			}
			if ((x < 2 || x >= a->c-2) && (y < 2 || y >= a->r-2)){
				char num[23];
				sprintf(num, "%.2e%+.2ei", a->raw_real[y*a->c+x], a->raw_imag[y*a->c+x]);
				printf("%23s", num);
			}else if ((x == 2 && (y < 3 || y >= a->r-2)) || (y == 2 && (x < 2 || x >= a->c-2))){
				printf("%23s", "...");
			}
			if (x == a->c-1 && (y < 3 || y >= a->r-2)){
				printf("]\n");
			}
		}
	}
	printf("\n");
}

void wekuaMatrixPrint(wmatrix *a){
	if (a == NULL){
		return;
	}else if(a->com){
		wekuaMatrixComplexPrint(a);
	}else{
		wekuaMatrixRealPrint(a);
	}
}

wmatrix *wekuaAllocMatrix(wekuaContext *ctx, uint32_t r, uint32_t c){
	if (ctx == NULL || r == 0 || c == 0){
		return NULL;
	}
	wmatrix *a = (wmatrix*) malloc(sizeof(wmatrix));
	a->com = 0;
	a->r = r;
	a->c = c;
	a->size = r*c;
	a->ctx = ctx;
	a->work_items[0] = getWI(a->size, ctx->max_work_group_size);
	uint64_t shape[2];
	shape[0] = r;
	shape[1] = c;
	getLWI(shape, &a->work_items[1], 2, a->ctx->max_work_group_size);

	int ret;
	a->real = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, sizeof(double)*a->size, NULL, &ret);
	if (ret != 0){
		printf("Failed to allocate new memory :-(\n");
		free(a);
		return NULL;
	}
	a->raw_real = NULL;

	a->imag = NULL;
	a->raw_imag = NULL;

	MapBufferMatrix(a);

	return a;
}

wmatrix *wekuaAllocComplexMatrix(wekuaContext *ctx, uint32_t r, uint32_t c){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	if (createComplexMatrix(a)){
		wekuaFreeMatrix(a);
		return NULL;
	}
	return a;
}

wmatrix *wekuaFillMatrix(wekuaContext *ctx, uint32_t r, uint32_t c, double alpha, double beta){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	if (a == NULL){
		printf("papu\n");
		return NULL;
	}
	if (beta != 0.0){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a);
			return NULL;
		}
	}
	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[0], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[0], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[0], 2, sizeof(double), &alpha);
	clSetKernelArg(a->ctx->kernels[0], 3, sizeof(double), &beta);
	runKernel(a->ctx->command_queue, a->ctx->kernels[0], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(a);
	return a;
}

wmatrix *wekuaMatrixRandn(wekuaContext *ctx, uint32_t r, uint32_t c, uint8_t com){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	if (a == NULL){
		return NULL;
	}
	if (com){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a);
			return NULL;
		}
	}

	cl_mem ran_r=NULL, ran_i=NULL;
	uint64_t *ran_r_m, *ran_i_m;
	int ret;
	
	ran_r = clCreateBuffer(a->ctx->ctx, CL_MEM_READ_WRITE, a->size*8, NULL, &ret);
	if (ret != 0){
		clReleaseMemObject(ran_r);
		return NULL;
	}
	ran_r_m = clEnqueueMapBuffer(a->ctx->command_queue, ran_r, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*8, 0, 0, NULL, NULL);
	getRandomBuffer(ran_r_m, a->size*8);
	clEnqueueUnmapMemObject(a->ctx->command_queue, ran_r, ran_r_m, 0, NULL, NULL);

	if (a->com){
		ran_i = clCreateBuffer(a->ctx->ctx, CL_MEM_READ_WRITE, a->size*8, NULL, &ret);
		if (ret != 0){
			clReleaseMemObject(ran_r);
			clReleaseMemObject(ran_i);
			return NULL;
		}
		ran_i_m = clEnqueueMapBuffer(a->ctx->command_queue, ran_i, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, a->size*8, 0, 0, NULL, NULL);
		getRandomBuffer(ran_i_m, a->size*8);
		clEnqueueUnmapMemObject(a->ctx->command_queue, ran_i, ran_i_m, 0, NULL, NULL);
	}

	UnmapBufferMatrix(a);
	clSetKernelArg(a->ctx->kernels[1], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[1], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[1], 2, sizeof(cl_mem), &ran_r);
	clSetKernelArg(a->ctx->kernels[1], 3, sizeof(cl_mem), &ran_i);
	clSetKernelArg(a->ctx->kernels[1], 4, 1, &a->com);
	runKernel(a->ctx->command_queue, a->ctx->kernels[1], 1, NULL, &a->size, a->work_items);
	MapBufferMatrix(a);

	clReleaseMemObject(ran_r);
	clReleaseMemObject(ran_i);

	return a;
}

wmatrix *wekuaMatrixFromBuffer(wekuaContext *ctx, uint32_t r, uint32_t c, void *rbuf, void *ibuf){
	wmatrix *a = wekuaAllocMatrix(ctx, r, c);
	memcpy(a->raw_real, rbuf, a->size*sizeof(double));
	if (ibuf != NULL){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(a);
			return NULL;
		}
		memcpy(a->raw_imag, ibuf, a->size*sizeof(double));
	}
	return a;
}

wmatrix *wekuaMatrixIden(wekuaContext *ctx, uint32_t c){
	wmatrix *i = wekuaFillMatrix(ctx, c, c, 0.0, 0.0);
	UnmapBufferMatrix(i);
	clSetKernelArg(i->ctx->kernels[2], 0, sizeof(cl_mem), &i->real);
	clSetKernelArg(i->ctx->kernels[2], 1, 4, &i->c);
	uint64_t wi, shape;
	shape = (uint64_t)i->c;
	wi = getWI(c, i->ctx->max_work_item_dimensions);
	runKernel(i->ctx->command_queue, i->ctx->kernels[2], 1, NULL, &shape, &wi);
	MapBufferMatrix(i);
	return i;
}

wmatrix *wekuaMatrixTrans(wmatrix *a){
	wmatrix *b = wekuaAllocMatrix(a->ctx, a->c, a->r);
	UnmapBufferMatrix(a);
	UnmapBufferMatrix(b);

	uint64_t shape[2];
	shape[0] = a->r;
	shape[1] = a->c;

	clSetKernelArg(a->ctx->kernels[3], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[3], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[3], 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[3], 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[3], 4, 4, &a->c);
	clSetKernelArg(a->ctx->kernels[3], 5, 4, &a->r);
	clSetKernelArg(a->ctx->kernels[3], 6, 1, &a->com);
	runKernel(a->ctx->command_queue, a->ctx->kernels[3], 2, NULL, shape, &a->work_items[1]);

	MapBufferMatrix(a);
	MapBufferMatrix(b);
	return b;
}

wmatrix *wekuaMatrixCopy(wmatrix *a){
	return wekuaMatrixFromBuffer(a->ctx, a->r, a->c, a->raw_real, a->raw_imag);
}

wmatrix *wekuaCutMatrix(wmatrix *a, uint32_t x, uint32_t w, uint32_t y, uint32_t h){
	wmatrix *b = wekuaAllocMatrix(a->ctx, h, w);
	if (w == 0 || h == 0 || x+w+1 > a->c || y+h+1 > a->r || w > a->c || h > a->r || b == NULL){
		return NULL;
	}
	uint64_t shape[2];
	shape[0] = h;
	shape[1] = w;

	UnmapBufferMatrix(a);
	UnmapBufferMatrix(b);
	clSetKernelArg(a->ctx->kernels[4], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[4], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[4], 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[4], 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[4], 4, 4, &b->c);
	clSetKernelArg(a->ctx->kernels[4], 5, 4, &x);
	clSetKernelArg(a->ctx->kernels[4], 6, 4, &y);
	clSetKernelArg(a->ctx->kernels[4], 7, 4, &a->c);
	clSetKernelArg(a->ctx->kernels[4], 8, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[4], 2, NULL, shape, &b->work_items[1]);

	MapBufferMatrix(a);
	MapBufferMatrix(b);
	return b;
}

wmatrix *wekuaMatrixResize(wmatrix *a, uint32_t r, uint32_t c){
	if (a == NULL){
		return NULL;
	}
	wmatrix *b = wekuaFillMatrix(a->ctx, r, c, 0.0, 0.0);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
			return NULL;
		}
	}

	UnmapBufferMatrix(a);
	UnmapBufferMatrix(b);

	uint64_t shape[2], wi[2];
	shape[0] = b->r;
	shape[1] = b->c;

	if (shape[0] > a->r){
		shape[0] = a->r;
	}
	if (shape[1] > a->c){
		shape[1] = a->c;
	}

	getLWI(shape, wi, 2, a->ctx->max_work_group_size);

	clSetKernelArg(a->ctx->kernels[19], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[19], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[19], 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[19], 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[19], 4, 4, &b->c);
	clSetKernelArg(a->ctx->kernels[19], 5, 4, &a->c);
	clSetKernelArg(a->ctx->kernels[19], 6, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[19], 2, NULL, shape, wi);

	MapBufferMatrix(a);
	MapBufferMatrix(b);

	return b;
}

wmatrix *wekuaMatrixProduct(wmatrix *a, wmatrix *b){
	if (a == NULL || b == NULL){
		return NULL;
	}else if (a->c != b->r){
		return NULL;
	}
	wmatrix *c = wekuaFillMatrix(a->ctx, a->r, b->c, 0.0, 0.0);
	if (a->com || b->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(c);
			return NULL;
		}
	}

	if (c == NULL){
		return NULL;
	}else if (a->com && b->com == 0){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(c);
			return NULL;
		}
	}else if (b->com && a->com == 0){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(c);
			return NULL;
		}
	}

	uint64_t shape[3], wi[3];
	shape[0] = a->r;
	shape[1] = b->c;
	shape[2] = a->c;
	wi[2] = 1;

	getLWI(shape, wi, 2, a->ctx->max_work_group_size);

	UnmapBufferMatrix(c);

	clSetKernelArg(a->ctx->kernels[6], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[6], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[6], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[6], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[6], 4, sizeof(cl_mem), &c->real);
	clSetKernelArg(a->ctx->kernels[6], 5, sizeof(cl_mem), &c->imag);
	clSetKernelArg(a->ctx->kernels[6], 6, sizeof(uint8_t), &a->com);
	clSetKernelArg(a->ctx->kernels[6], 7, sizeof(uint32_t), &a->c);
	clSetKernelArg(a->ctx->kernels[6], 8, sizeof(uint32_t), &b->c);

	runKernel(a->ctx->command_queue, a->ctx->kernels[6], 3, NULL, shape, wi);

	MapBufferMatrix(c);
	return c;
}

void Axpy(wmatrix *a, wmatrix *b, double alpha){
	if (a == NULL || b == NULL){
		return;
	}else if(a->c != b->c || a->r != b->r){
		return;
	}

	if (a->com && b->com == 0){
		if (createComplexMatrix(b)){
			return;
		}
	}else if (b->com && a->com == 0){
		if (createComplexMatrix(a)){
			return;
		}
	}

	UnmapBufferMatrix(a);
	UnmapBufferMatrix(b);

	clSetKernelArg(a->ctx->kernels[5], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[5], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[5], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[5], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[5], 4, 4, &a->c);
	clSetKernelArg(a->ctx->kernels[5], 5, 1, &a->com);
	clSetKernelArg(a->ctx->kernels[5], 6, sizeof(double), &alpha);

	runKernel(a->ctx->command_queue, a->ctx->kernels[5], 1, NULL, &a->size, a->work_items);

	MapBufferMatrix(a);
	MapBufferMatrix(b);
}

void wekuaMatrixAdd(wmatrix *a, wmatrix *b){
	Axpy(a, b, 1.0);
}

void wekuaMatrixSub(wmatrix *a, wmatrix *b){
	Axpy(a, b, -1.0);
}

void wekuaMatrixDot(wmatrix *a, double alpha, double beta){
	if (a == NULL){
		return;
	}

	if (beta != 0.0 && a->com == 0){
		if (createComplexMatrix(a)){
			return;
		}
	}

	UnmapBufferMatrix(a);

	clSetKernelArg(a->ctx->kernels[13], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[13], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[13], 2, 1, &a->com);
	clSetKernelArg(a->ctx->kernels[13], 3, sizeof(double), &alpha);
	clSetKernelArg(a->ctx->kernels[13], 4, sizeof(double), &beta);

	runKernel(a->ctx->command_queue, a->ctx->kernels[13], 1, NULL, &a->size, a->work_items);

	MapBufferMatrix(a);
}

void wekuaMatrixAbs(wmatrix *a){
	if (a == NULL){
		return;
	}

	UnmapBufferMatrix(a);

	uint64_t shape[2];
	shape[0] = a->r;
	shape[1] = a->c;

	clSetKernelArg(a->ctx->kernels[14], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[14], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[14], 2, 4, &a->c);
	clSetKernelArg(a->ctx->kernels[14], 3, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[14], 2, NULL, shape, &a->work_items[1]);

	MapBufferMatrix(a);

	removeComplexMatrix(a);
}

void wekuaMatrixAbsdiff(wmatrix *a, wmatrix *b){
	if (a == NULL || b == NULL){
		return;
	}
	wekuaMatrixSub(a, b);
	wekuaMatrixAbs(a);
}

wmatrix *wekuaMatrixDiag(wmatrix *a){
	if (a == NULL){
		return NULL;
	}else if (a->r != a->c){
		return NULL;
	}
	wmatrix *b = wekuaAllocMatrix(a->ctx, 1, a->c);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
			return NULL;
		}
	}

	UnmapBufferMatrix(b);

	clSetKernelArg(a->ctx->kernels[15], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[15], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[15], 2, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[15], 3, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[15], 4, sizeof(uint32_t), &a->c);
	clSetKernelArg(a->ctx->kernels[15], 5, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[15], 1, NULL, &b->size, b->work_items);

	MapBufferMatrix(b);
	return b;
}

void wekuaMatrixSum(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}

	double re=0.0, im=0.0;
	uint64_t shape[2], wi[2];

	wmatrix *b;

	if (a->r == 1 || a->c == 1){
		b = wekuaMatrixCopy(a);
		goto Sumstetwo;
	}

	b = wekuaFillMatrix(a->ctx, 1, a->r, 0.0, 0.0);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
			return;
		}
	}

	UnmapBufferMatrix(a);
	UnmapBufferMatrix(b);

	shape[0] = a->r;
	shape[1] = a->c;
	wi[0] = b->work_items[0];
	wi[1] = 1;

	clSetKernelArg(a->ctx->kernels[16], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[16], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[16], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[16], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[16], 4, 4, &a->c);
	clSetKernelArg(a->ctx->kernels[16], 5, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[16], 2, NULL, shape, wi);

	MapBufferMatrix(a);
	MapBufferMatrix(b);

	Sumstetwo:
	for (uint64_t i=0; i<b->size; i++){
		re += b->raw_real[i];
		if (a->com){
			im += b->raw_imag[i];
		}
	}

	if (real != NULL){
		real[0] = re;
	}
	if (a->com && imag != NULL){
		imag[0] = im;
	}

	wekuaFreeMatrix(b);
}

void wekuaMatrixMul(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}

	double re=1.0, im=1.0;
	uint64_t shape[2], wi[2];

	wmatrix *b;

	if (a->r == 1 || a->c == 1){
		b = wekuaMatrixCopy(a);
		goto Sumstetwo;
	}

	b = wekuaFillMatrix(a->ctx, 1, a->r, 1.0, 0.0);

	if (a->com){
		if (createComplexMatrix(b)){
			wekuaFreeMatrix(b);
			return;
		}
	}

	UnmapBufferMatrix(a);
	UnmapBufferMatrix(b);

	shape[0] = a->r;
	shape[1] = a->c;
	wi[0] = b->work_items[0];
	wi[1] = 1;

	clSetKernelArg(a->ctx->kernels[17], 0, sizeof(cl_mem), &a->real);
	clSetKernelArg(a->ctx->kernels[17], 1, sizeof(cl_mem), &a->imag);
	clSetKernelArg(a->ctx->kernels[17], 2, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[17], 3, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[17], 4, 4, &a->c);
	clSetKernelArg(a->ctx->kernels[17], 5, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[17], 2, NULL, shape, wi);

	MapBufferMatrix(a);
	MapBufferMatrix(b);

	Sumstetwo:
	for (uint64_t i=0; i<b->size; i++){
		if (a->com){
			re = b->raw_real[i]*re - b->raw_imag[i]*im;
			im = b->raw_real[i]*im + b->raw_imag[i]*re;
		}else{
			re *= b->raw_real[i];
		}
	}

	if (real != NULL){
		real[0] = re;
	}
	if (a->com && imag != NULL){
		imag[0] = im;
	}

	wekuaFreeMatrix(b);
}

void wekuaMatrixMean(wmatrix *a, double *real, double *imag){
	wmatrix *b = wekuaMatrixCopy(a);
	double n = b->size;
	n = 1/n;
	wekuaMatrixDot(b, n, 0.0);
	wekuaMatrixSum(b, real, imag);
	wekuaFreeMatrix(b);
}

void wekuaMatrixNorm(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}
	wmatrix *b = wekuaMatrixCopy(a);

	UnmapBufferMatrix(b);

	clSetKernelArg(a->ctx->kernels[18], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[18], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[18], 2, 1, &b->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[18], 1, NULL, &a->size, a->work_items);

	MapBufferMatrix(b);

	wekuaMatrixSum(b, real, imag);

	double ang, n;

	if (a->com){
		if (real[0] == 0){
			ang = CL_M_PI_2;
		}else{
			ang = tanh(imag[0]/real[0])/2;	
		}
		n = sqrt(real[0]*real[0]+imag[0]*imag[0]);
		real[0] = n*cos(ang);
		imag[0] = n*sin(ang);
	}else{
		real[0] = sqrt(real[0]);
	}
}

void wekuaMatrixDet(wmatrix *a, double *real, double *imag){
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}else if (a->r != a->c){
		return;
	}

	wmatrix *b = wekuaMatrixCopy(a);
	wmatrix *c = wekuaFillMatrix(a->ctx, a->r, a->c, 1.0, 0.0);
	if (b->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(c);
			wekuaFreeMatrix(b);
			return;
		}
	}

	uint64_t shape[2], wi[2];
	shape[0] = a->r;
	shape[1] = a->r;
	wi[0] = 1;
	wi[1] = getWI(a->r, a->ctx->max_work_group_size);

	UnmapBufferMatrix(b);
	UnmapBufferMatrix(c);

	clSetKernelArg(a->ctx->kernels[20], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[20], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[20], 2, sizeof(cl_mem), &c->real);
	clSetKernelArg(a->ctx->kernels[20], 3, sizeof(cl_mem), &c->imag);
	clSetKernelArg(a->ctx->kernels[20], 4, 4, &b->c);
	clSetKernelArg(a->ctx->kernels[20], 5, 1, &b->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[20], 2, NULL, shape, wi);

	MapBufferMatrix(c);

	wekuaMatrixMul(c, real, imag);

	wekuaFreeMatrix(b);
	wekuaFreeMatrix(c);
}

wmatrix *wekuaMatrixInv(wmatrix *a){
	if (a == NULL){
		return NULL;
	}
	double det_r=0.0, det_i=0.0;
	wekuaMatrixDet(a, &det_r, &det_i);
	if (det_r == 0.0 && det_i == 0.0){
		return NULL;
	}
	wmatrix *b, *c;
	b = wekuaMatrixCopy(a);
	c = wekuaMatrixIden(a->ctx, a->r);
	if (a->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(b);
			wekuaFreeMatrix(c);
			return NULL;
		}
	}
	if (b == NULL || c == NULL){
		wekuaFreeMatrix(b);
		wekuaFreeMatrix(c);
		return NULL;
	}
	uint8_t otherm=1;

	UnmapBufferMatrix(b);
	UnmapBufferMatrix(c);
	uint64_t shape[2], wi[2];
	shape[0] = a->r;
	shape[1] = a->r;
	wi[0] = 1;
	wi[1] = getWI(a->r, a->ctx->max_work_group_size);
	for (uint8_t t=0; t<2; t++){
		clSetKernelArg(a->ctx->kernels[21], 0, sizeof(cl_mem), &b->real);
		clSetKernelArg(a->ctx->kernels[21], 1, sizeof(cl_mem), &b->imag);
		clSetKernelArg(a->ctx->kernels[21], 2, sizeof(cl_mem), &c->real);
		clSetKernelArg(a->ctx->kernels[21], 3, sizeof(cl_mem), &c->imag);
		clSetKernelArg(a->ctx->kernels[21], 4, 4, &a->r);
		clSetKernelArg(a->ctx->kernels[21], 5, 1, &a->com);
		clSetKernelArg(a->ctx->kernels[21], 6, 1, &otherm);
		clSetKernelArg(a->ctx->kernels[21], 7, 1, &t);

		runKernel(a->ctx->command_queue, a->ctx->kernels[21], 2, NULL, shape, wi);
	}
	clSetKernelArg(a->ctx->kernels[22], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[22], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[22], 2, sizeof(cl_mem), &c->real);
	clSetKernelArg(a->ctx->kernels[22], 3, sizeof(cl_mem), &c->imag);
	clSetKernelArg(a->ctx->kernels[22], 4, 4, &a->r);
	clSetKernelArg(a->ctx->kernels[22], 5, 1, &a->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[22], 2, NULL, shape, &a->work_items[1]);

	MapBufferMatrix(c);
	wekuaFreeMatrix(b);
	return c;
}

wmatrix *wekuaMatrixSolve(wmatrix *a, wmatrix *b){
	wmatrix *c = wekuaMatrixInv(a);
	wmatrix *d = wekuaMatrixProduct(c, b);
	wekuaFreeMatrix(c);
	return d;
}

uint32_t wekuaMatrixRang(wmatrix *a){
	if (a == NULL){
		return 0;
	}
	wmatrix *b, *c;
	b = wekuaMatrixCopy(a);
	uint32_t rang=0;
	uint8_t otherm=0, t=0;
	uint64_t shape[2], wi[2];
	shape[0] = a->r;
	shape[1] = a->r;
	if (a->r > a->c){
		shape[0] = a->c;
	}
	wi[0] = 1;
	wi[1] = getWI(shape[1], a->ctx->max_work_group_size);
	getLWI(shape, wi, 2, a->ctx->max_work_group_size);
	cl_mem nullm = NULL;

	UnmapBufferMatrix(b);

	clSetKernelArg(a->ctx->kernels[21], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[21], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[21], 2, sizeof(cl_mem), &nullm);
	clSetKernelArg(a->ctx->kernels[21], 3, sizeof(cl_mem), &nullm);
	clSetKernelArg(a->ctx->kernels[21], 4, 4, &a->c);
	clSetKernelArg(a->ctx->kernels[21], 5, 1, &a->com);
	clSetKernelArg(a->ctx->kernels[21], 6, 1, &otherm);
	clSetKernelArg(a->ctx->kernels[21], 7, 1, &t);

	runKernel(a->ctx->command_queue, a->ctx->kernels[21], 2, NULL, shape, wi);

	wekuaMatrixAbs(b);

	c = wekuaFillMatrix(a->ctx, 1, b->r, 0.0, 0.0);
	if (a->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(b);
			wekuaFreeMatrix(c);
			return 0;
		}
	}

	shape[0] = b->r;
	shape[1] = b->c;
	wi[0] = c->work_items[0];
	wi[1] = 1;
	
	UnmapBufferMatrix(c);
	UnmapBufferMatrix(b);

	clSetKernelArg(a->ctx->kernels[16], 0, sizeof(cl_mem), &b->real);
	clSetKernelArg(a->ctx->kernels[16], 1, sizeof(cl_mem), &b->imag);
	clSetKernelArg(a->ctx->kernels[16], 2, sizeof(cl_mem), &c->real);
	clSetKernelArg(a->ctx->kernels[16], 3, sizeof(cl_mem), &c->imag);
	clSetKernelArg(a->ctx->kernels[16], 4, 4, &b->c);
	clSetKernelArg(a->ctx->kernels[16], 5, 1, &b->com);

	runKernel(a->ctx->command_queue, a->ctx->kernels[16], 2, NULL, shape, wi);

	wekuaFreeMatrix(b);
	wekuaMatrixAbs(c);

	for (uint32_t r=0; r<c->c; r++){
		if (c->raw_real[r] != 0){
			rang++;
		}
	}
	wekuaFreeMatrix(c);
	return rang;
}

wmatrix *wekuaMatrixPinv(wmatrix *a){
	if (a == NULL){
		return NULL;
	}
	wmatrix *pinv=NULL, *ta=NULL, *te=NULL, *ti=NULL;
	ta = wekuaMatrixTrans(a);
	uint32_t rang = wekuaMatrixRang(a);
	if (rang == a->r){
		te = wekuaMatrixProduct(a, ta);
		ti = wekuaMatrixInv(te);
		pinv = wekuaMatrixProduct(ta, ti);
	}else if (rang == a->c){
		te = wekuaMatrixProduct(ta, a);
		ti = wekuaMatrixInv(te);
		pinv = wekuaMatrixProduct(ti, ta);
	}
	wekuaFreeMatrix(ta);
	wekuaFreeMatrix(te);
	wekuaFreeMatrix(ti);
	return pinv;
}

void wekuaMatrixTrace(wmatrix *a, double *real, double *imag){
	wmatrix *b = wekuaMatrixDiag(a);
	wekuaMatrixSum(b, real, imag);
}

wmatrix *wekuaMatrixPoly(wmatrix *a){
	if (a == NULL){
		return NULL;
	}else if (a->r != a->c){
		return NULL;
	}
	wmatrix *c, **b, *i;
	c = wekuaAllocMatrix(a->ctx, 1, a->r+1);
	if (a->com){
		if (createComplexMatrix(c)){
			wekuaFreeMatrix(c);
			return NULL;
		}
	}
	b = (wmatrix**) malloc(sizeof(wmatrix*)*2);
	b[0] = wekuaFillMatrix(a->ctx, a->r, a->c, 0.0, 0.0);
	c->raw_real[a->r] = 1.0;
	for (uint32_t x=1; x<=a->r; x++){
		i = wekuaMatrixIden(a->ctx, a->r);
		b[1] = wekuaMatrixProduct(a, b[0]);
		if (a->com){
			wekuaMatrixDot(i, c->raw_real[a->r-x+1], c->raw_imag[a->r-x+1]);
		}else{
			wekuaMatrixDot(i, c->raw_real[a->r-x+1], 0.0);
		}
		wekuaMatrixAdd(b[1], i);
		wekuaFreeMatrix(i);
		i = wekuaMatrixProduct(a, b[1]);
		if (a->com){
			wekuaMatrixTrace(i, &c->raw_real[a->r-x], &c->raw_imag[a->r-x]);
			c->raw_imag[a->r-x] /= -1.0*x;
		}else{
			wekuaMatrixTrace(i, &c->raw_real[a->r-x], NULL);
			c->raw_real[a->r-x] /= -1.0*x;
		}
		wekuaFreeMatrix(i);
		wekuaFreeMatrix(b[0]);
		b[0] = wekuaMatrixCopy(b[1]);
		wekuaFreeMatrix(b[1]);
	}
	wekuaFreeMatrix(b[0]);
	free(b);
	return c;
}

// wmatrix *wekuaMatrixRoot(wmatrix *a){
// 	if (a == NULL){
// 		return NULL;
// 	}
// 	wmatrix *coeff = wekuaAllocMatrix(a->ctx, a->r, a->c-1);

// 	return coeff;
// }
#include "wekua.h"
#include <CL/cl.h>
#include <bits/stdint-uintn.h>
#include <stdint.h>
#include <stdio.h>

#define GEMM_KERNELS_NUM 4

const uint64_t zero_gemm = 0;

void getLWI(uint64_t *x, uint64_t *y, uint32_t si, uint64_t max);

char *getKernelData(const char *name, long *size);

const char kernels_gemm[GEMM_KERNELS_NUM][38] = {
	"/usr/lib/wekua_kernels/gemm_fast_1.cl",
	"/usr/lib/wekua_kernels/gemm_fast_2.cl",
	"/usr/lib/wekua_kernels/gemm_fast_3.cl",
	"/usr/lib/wekua_kernels/gemm_fast_4.cl"
};

const char kernel_gemm_name[GEMM_KERNELS_NUM][12] = {
	"gemm_fast_1", "gemm_fast_2", "gemm_fast_3",
	"gemm_fast_4"
};

void show_program_build_log(cl_program pro, cl_device_id dev, int ret, uint8_t ker){
	uint64_t size;
	char *error;

	clGetProgramBuildInfo(pro, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
	error = malloc(size);
	clGetProgramBuildInfo(pro, dev, CL_PROGRAM_BUILD_LOG, size, error, NULL);
	printf("%s\n", error);
	printf("\nSize: %lu\n", size);
	free(error);

	printf("Return code: %d\nKernel: %s\n", ret, kernel_gemm_name[ker]);
}

uint8_t compileGemmKernels(wekuaContext ctx, uint8_t dtype){
	if (ctx == NULL || dtype > WEKUA_DTYPE_DOUBLE) return 1;

	uint32_t vl = ctx->vector_width[dtype], sdu = ctx->num_sub_dev;
	uint64_t size;
	int ret;
	char *source, args[40];
	cl_context context = ctx->ctx;

	if (ctx->gemm_kernels == NULL){
		ctx->gemm_kernels = (cl_kernel**) calloc(4, sizeof(cl_kernel*));
		if (ctx->gemm_kernels == NULL) return 1;
	}

	if (ctx->gemm_programs == NULL){
		ctx->gemm_programs = (cl_program**) calloc(4, sizeof(cl_program*));
		if (ctx->gemm_programs == NULL) return 1;
	}

	// Alloc cl_kernel and cl_program

	check_gemm_com_1:
	if (ctx->gemm_programs[0] == NULL){
		ctx->gemm_programs[0] = (cl_program*) calloc(10, sizeof(cl_program));
		if (ctx->gemm_programs[0] == NULL) return 1;
	}
	if (ctx->gemm_kernels[0] == NULL){
		ctx->gemm_kernels[0] = (cl_kernel*) calloc(10, sizeof(cl_kernel));
		if (ctx->gemm_kernels[0] == NULL) return 1;
	}
	if (ctx->gemm_kernels[0][dtype] == NULL || ctx->gemm_programs[0][dtype] == NULL) goto kernel_gemm_fast_1;


	if (ctx->gemm_programs[1] == NULL){
		ctx->gemm_programs[1] = (cl_program*) calloc(10, sizeof(cl_program));
		if (ctx->gemm_programs[1] == NULL) return 1;
	}
	if (ctx->gemm_kernels[1] == NULL){
		ctx->gemm_kernels[1] = (cl_kernel*) calloc(10, sizeof(cl_kernel));
		if (ctx->gemm_kernels[1] == NULL) return 1;
	}
	if (ctx->gemm_kernels[1][dtype] == NULL || ctx->gemm_programs[1][dtype] == NULL) goto kernel_gemm_fast_2;

	if (ctx->gemm_programs[2] == NULL){
		ctx->gemm_programs[2] = (cl_program*) calloc(70, sizeof(cl_program));
		if (ctx->gemm_programs[2] == NULL) return 1;
	}
	if (ctx->gemm_kernels[2] == NULL){
		ctx->gemm_kernels[2] = (cl_kernel*) calloc(70, sizeof(cl_kernel));
		if (ctx->gemm_kernels[2] == NULL) return 1;
	}
	goto kernel_gemm_fast_3;

	check_gemm_com_2:
	if (ctx->gemm_programs[3] == NULL){
		ctx->gemm_programs[3] = (cl_program*) calloc(10, sizeof(cl_program));
		if (ctx->gemm_programs[3] == NULL) return 1;
	}
	if (ctx->gemm_kernels[3] == NULL){
		ctx->gemm_kernels[3] = (cl_kernel*) calloc(10, sizeof(cl_kernel));
		if (ctx->gemm_kernels[3] == NULL) return 1;
	}
	if (ctx->gemm_kernels[3][dtype] == NULL || ctx->gemm_programs[3][dtype] == NULL) goto kernel_gemm_fast_4;

	goto kernel_gemm_fast_end;

	kernel_gemm_fast_1:
	sprintf(args, "-Dwidth=%d -Ddtype=%d", vl, dtype);
	source = getKernelData(kernels_gemm[0], (long*)&size);
	if (source == NULL) return 1;

	if (ctx->gemm_programs[0][dtype] == NULL){
		ctx->gemm_programs[0][dtype] = clCreateProgramWithSource(context, 1, (const char**)&source, &size, &ret);
		if (ret != CL_SUCCESS) return 1;
		ret = clBuildProgram(ctx->gemm_programs[0][dtype], 1, &ctx->dev, args, NULL, NULL);
	}
	free(source);

	if (ret != CL_SUCCESS){
		show_program_build_log(ctx->gemm_programs[0][dtype], ctx->dev, ret, 0);
		ctx->gemm_programs[0][dtype] = NULL;
		return 1;
	}

	if (ctx->gemm_kernels[0][dtype] == NULL){
		ctx->gemm_kernels[0][dtype] = clCreateKernel(ctx->gemm_programs[0][dtype], kernel_gemm_name[0], &ret);
		if (ret != CL_SUCCESS) return 1;
	}

	goto check_gemm_com_1;

	kernel_gemm_fast_2:
	sprintf(args, "-Dwidth=%d -Ddtype=%d", vl, dtype);
	source = getKernelData(kernels_gemm[1], (long*)&size);
	if (source == NULL) return 1;

	if (ctx->gemm_programs[1][dtype] == NULL){
		ctx->gemm_programs[1][dtype] = clCreateProgramWithSource(context, 1, (const char**)&source, &size, &ret);
		ret = clBuildProgram(ctx->gemm_programs[1][dtype], 1, &ctx->dev, args, NULL, NULL);
	}
	free(source);

	if (ret != CL_SUCCESS){
		show_program_build_log(ctx->gemm_programs[1][dtype], ctx->dev, ret, 1);
		ctx->gemm_programs[1][dtype] = NULL;
		return 1;
	}

	if (ctx->gemm_kernels[1][dtype] == NULL){
		ctx->gemm_kernels[1][dtype] = clCreateKernel(ctx->gemm_programs[1][dtype], kernel_gemm_name[1], &ret);
		if (ret != CL_SUCCESS) return 1;
	}

	goto check_gemm_com_1;

	kernel_gemm_fast_3:
	ret = CL_SUCCESS;
	source = getKernelData(kernels_gemm[2], (long*)&size);
	if (source == NULL) return 1;
	for (uint32_t j=0; j<7; j++){
		memset(args, 0, 40);
		sprintf(args, "-Dwidth=%d -Ddtype=%d -Dks=%d", vl, dtype, j);

		if (ctx->gemm_programs[2][j*10+dtype] == NULL){
			ctx->gemm_programs[2][j*10+dtype] = clCreateProgramWithSource(context, 1, (const char**)&source, &size, &ret);
			ret = clBuildProgram(ctx->gemm_programs[2][j*10+dtype], 1, &ctx->subdevice[j%sdu], args, NULL, NULL);
		}

		if (ret != CL_SUCCESS){
			show_program_build_log(ctx->gemm_programs[2][j*10+dtype], ctx->subdevice[j%sdu], ret, 2);
			ctx->gemm_programs[2][j*10+dtype] = NULL;
			return 1;
		}

		if (ctx->gemm_kernels[2][j*10+dtype] == NULL){
			ctx->gemm_kernels[2][j*10+dtype] = clCreateKernel(ctx->gemm_programs[2][j*10+dtype], kernel_gemm_name[2], &ret);
			if (ret != CL_SUCCESS) return 1;
		}
	}
	free(source);

	goto check_gemm_com_2;

	kernel_gemm_fast_4:
	sprintf(args, "-Dwidth=%d -Ddtype=%d", vl, dtype);
	source = getKernelData(kernels_gemm[3], (long*)&size);
	if (source == NULL) return 1;

	if (ctx->gemm_programs[3][dtype] == NULL){
		ctx->gemm_programs[3][dtype] = clCreateProgramWithSource(context, 1, (const char**)&source, &size, &ret);
		if (ret != CL_SUCCESS) return 1;
		ret = clBuildProgram(ctx->gemm_programs[3][dtype], 1, &ctx->dev, args, NULL, NULL);
	}
	free(source);

	if (ret != CL_SUCCESS){
		show_program_build_log(ctx->gemm_programs[3][dtype], ctx->dev, ret, 3);
		ctx->gemm_programs[3][dtype] = NULL;
		return 1;
	}

	if (ctx->gemm_kernels[3][dtype] == NULL){
		ctx->gemm_kernels[3][dtype] = clCreateKernel(ctx->gemm_programs[3][dtype], kernel_gemm_name[3], &ret);
		if (ret != CL_SUCCESS) return 1;
	}

	kernel_gemm_fast_end:
	return 0;
}

// Kernels exec
int runGemmFast_12(cl_command_queue cmd, cl_kernel kernel, void *com, uint64_t *shape, uint64_t *local,
	wmatrix x, wmatrix *alpha, cl_event *event
){
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &x->imag);
	for (uint32_t j=0; j<5; j++){
		clSetKernelArg(kernel, 2 + j*2, sizeof(cl_mem), &alpha[j]->real);
		clSetKernelArg(kernel, 2 + j*2 + 1, sizeof(cl_mem), &alpha[j]->imag);
	}
	clSetKernelArg(kernel, 12, 8, &x->col);
	clSetKernelArg(kernel, 13, 8, &alpha[0]->col);
	clSetKernelArg(kernel, 14, 1, com);

	return clEnqueueNDRangeKernel(cmd, kernel, 2, NULL, shape, local, 0, NULL, event);
}

int runGemmFast_3(cl_command_queue cmd, cl_kernel kernel, void *com, void *m, uint64_t *shape, uint64_t *local,
	wmatrix alpha, wmatrix beta, wmatrix d, cl_event *be, cl_event *event
){
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &alpha->real);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &alpha->imag);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &beta->real);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &beta->imag);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &d->real);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &d->imag);

	clSetKernelArg(kernel, 6, 8, &alpha->col);
	clSetKernelArg(kernel, 7, 8, &beta->col);
	clSetKernelArg(kernel, 8, 8, &d->col);
	clSetKernelArg(kernel, 9, 8, m);
	clSetKernelArg(kernel, 10, 1, com);

	return clEnqueueNDRangeKernel(cmd, kernel, 2, NULL, shape, local, 2, be, event);
}

int runGemmFast_4(cl_command_queue cmd, cl_kernel kernel, void *com, uint64_t *shape, uint64_t *local, wmatrix c,
	wmatrix *d, void *ralpha, void *ialpha, void *rbeta, void *ibeta, uint32_t lm, cl_event *e){
	for (uint32_t j=0; j<7; j++){
		clSetKernelArg(kernel, j*2, sizeof(cl_mem), &d[j]->real);
		clSetKernelArg(kernel, j*2 + 1, sizeof(cl_mem), &d[j]->imag);
	}

	clSetKernelArg(kernel, 14, lm, ralpha);
	clSetKernelArg(kernel, 15, lm, ialpha);
	clSetKernelArg(kernel, 16, lm, rbeta);
	clSetKernelArg(kernel, 17, lm, ibeta);

	clSetKernelArg(kernel, 18, sizeof(cl_mem), &c->real);
	clSetKernelArg(kernel, 19, sizeof(cl_mem), &c->imag);
	clSetKernelArg(kernel, 20, 8, &c->col);
	clSetKernelArg(kernel, 21, 8, &d[0]->col);
	clSetKernelArg(kernel, 22, 1, com);

	return clEnqueueNDRangeKernel(cmd, kernel, 2, NULL, shape, local, 0, NULL, e);
}

int wekuaBlasFastGemm(
	void *ralpha, void *ialpha, uint8_t a_trans, wmatrix a, uint8_t b_trans, wmatrix b,
	void *rbeta, void *ibeta, wmatrix c, uint32_t nw, cl_event *be
){
	wekuaContext ctx = a->ctx;
	cl_command_queue cmd = ctx->command_queue;
	cl_command_queue *gemm_cmd = ctx->gemm_cmd;
	cl_kernel **kernel = ctx->gemm_kernels;
	cl_event e[30];

	int ret = CL_SUCCESS;
	uint8_t dtype = a->dtype, com = a->com|b->com|c->com;
	uint32_t evn = 0, sdu = ctx->num_sub_dev;
	uint64_t shape[6], wi[8], m, max;

	wmatrix x, y;
	wmatrix beta[5], alpha[5], d[7];

	if (compileGemmKernels(ctx, dtype)) return CL_COMPILE_PROGRAM_FAILURE;

	if (ralpha == NULL) ralpha = (uint64_t*)&zero_gemm;
	if (ialpha == NULL) ialpha = (uint64_t*)&zero_gemm;
	if (rbeta == NULL) rbeta = (uint64_t*)&zero_gemm;
	if (ibeta == NULL) ibeta = (uint64_t*)&zero_gemm;
	clWaitForEvents(nw, be);

	if (a_trans){
		x = wekuaMatrixTrans(a, 0, NULL, e);
		clWaitForEvents(1, e);
		clReleaseEvent(e[0]);
	}else { x = a; }

	if (b_trans) y = b;
	else{
		y = wekuaMatrixTrans(b, 0, NULL, e);
		clWaitForEvents(1, e);
		clReleaseEvent(e[0]);
	}

	if (x == NULL || y == NULL) goto gemm_fast_end;

	if (com){
		if (createComplexMatrix(x)|createComplexMatrix(y)|createComplexMatrix(c)) goto gemm_fast_end;
	}

	max = ctx->max_work_group_size;
	
	shape[0] = c->col_g/2;
	shape[1] = c->row_g/2;

	m = x->col_g/2;

	shape[2] = shape[0];
	shape[4] = shape[1];

	shape[3] = m;
	shape[5] = m;

	m /= 2;

	getLWI(shape, wi, 2, max);
	getLWI(&shape[2], &wi[2], 2, max);
	getLWI(&shape[4], &wi[4], 2, max);
	getLWI(shape, &wi[6], 1, max);
	getLWI(shape, &wi[7], 1, max);

	for (uint8_t j=0; j<7; j++){
		d[j] = wekuaAllocMatrix(ctx, shape[0], shape[1], dtype);
		if (j < 5){
			alpha[j] = wekuaMatrixEmpty(ctx, shape[2], shape[3], dtype);
			beta[j] = wekuaMatrixEmpty(ctx, shape[4], shape[5], dtype);
			if (alpha[j] == NULL || beta[j] == NULL) goto gemm_fast_end;
			if (com){
				if (createComplexMatrix(alpha[j])|createComplexMatrix(beta[j])) goto gemm_fast_end;
			}
		}
		if (d[j] == NULL) goto gemm_fast_end;
		if (com){
			if (createComplexMatrix(d[j])) goto gemm_fast_end;
		}
	}

	ret = runGemmFast_12(cmd, kernel[0][dtype], &com, &shape[2], &wi[2], x, alpha, e);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;

	ret = runGemmFast_12(cmd, kernel[1][dtype], &com, &shape[4], &wi[4], y, beta, &e[1]);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;

	ret = runGemmFast_3(gemm_cmd[0], kernel[2][dtype], &com, &m, shape, wi, alpha[0], beta[0], d[0], e, &e[evn]);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;

	ret = runGemmFast_3(gemm_cmd[1%sdu], kernel[2][10+dtype], &com, &m, shape, wi, alpha[1], y, d[1], e, &e[evn]);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;

	ret = runGemmFast_3(gemm_cmd[2%sdu], kernel[2][20+dtype], &com, &m, shape, wi, x, beta[1], d[2], e, &e[evn]);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;

	ret = runGemmFast_3(gemm_cmd[3%sdu], kernel[2][30+dtype], &com, &m, shape, wi, x, beta[2], d[3], e, &e[evn]);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;
	
	ret = runGemmFast_3(gemm_cmd[4%sdu], kernel[2][40+dtype], &com, &m, shape, wi, alpha[2], y, d[4], e, &e[evn]);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;

	ret = runGemmFast_3(gemm_cmd[5%sdu], kernel[2][50+dtype], &com, &m, shape, wi, alpha[3], beta[3], d[5], e, &e[evn]);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;

	ret = runGemmFast_3(gemm_cmd[6%sdu], kernel[2][60+dtype], &com, &m, shape, wi, alpha[4], beta[4], d[6], e, &e[evn]);
	if (ret == CL_SUCCESS) evn++;
	else goto gemm_fast_end;

	clWaitForEvents(evn, e);
	
	ret = runGemmFast_4(cmd, kernel[3][dtype], &com, shape, wi, c, d, ralpha, ialpha, rbeta, ibeta, ctx->dtype_length[dtype], &e[evn]);

	gemm_fast_end:
	if (evn > 0) clWaitForEvents(1, &e[evn]);
	for (uint32_t j=0; j<evn; j++) clReleaseEvent(e[j]);

	if (x != a) wekuaFreeMatrix(x, 0, NULL);
	if (y != b) wekuaFreeMatrix(y, 0, NULL);

	for (uint8_t j=0; j<7; j++){
		if (j < 5){
			wekuaFreeMatrix(alpha[j], 0, NULL);
			wekuaFreeMatrix(beta[j], 0, NULL);
		}
		wekuaFreeMatrix(d[j], 0, NULL);
	}

	return ret;
}
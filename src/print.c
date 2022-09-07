#include "../headers/matrix.h"
#include <stdio.h>

void wekuaGetValueFromMatrix(wmatrix a, uint64_t y, uint64_t x, void *real, void *imag, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (a == NULL || (real == NULL && imag == NULL)) return;

	uint64_t len_dtype = a->ctx->dtype_length[a->dtype];
	uint64_t posi = y*a->col*len_dtype + x*len_dtype;

	wekuaContext ctx = a->ctx;

	cl_event e[2];
	uint32_t nevents = 0;

	if (real){
		clEnqueueReadBuffer(ctx->command_queue, a->real, CL_FALSE, posi, len_dtype, real, 0, NULL, &e[nevents++]);
	}
	if (imag && a->com){
		clEnqueueReadBuffer(ctx->command_queue, a->imag, CL_FALSE, posi, len_dtype, imag, 0, NULL, &e[nevents++]);
	}
	clWaitForEvents(nevents, e);
	for (uint32_t x=0; x<nevents; x++) clReleaseEvent(e[x]);
}

void wekuaPutValueToMatrix(wmatrix a, uint64_t y, uint64_t x, void *real, void *imag, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (a == NULL){
		return;
	}else if (real == NULL && imag == NULL){
		return;
	}

	uint64_t len_dtype = a->ctx->dtype_length[a->dtype];
	uint64_t posi = y*a->col*len_dtype + x*len_dtype, zero = 0;

	wekuaContext ctx = a->ctx;

	cl_event e[2];
	uint32_t nevents = 0;

	if (real){
		clEnqueueWriteBuffer(ctx->command_queue, a->real, CL_FALSE, posi, len_dtype, real, 0, NULL, &e[nevents++]);
	}
	if (imag && a->com){
		clEnqueueWriteBuffer(ctx->command_queue, a->imag, CL_FALSE, posi, len_dtype, imag, 0, NULL, &e[nevents++]);
	}
	clWaitForEvents(nevents, e);
	for (uint32_t x=0; x<nevents; x++) clReleaseEvent(e[x]);
}

const char dtype_text[][15] = {
	"int8", "uint8",
	"int16", "uint16",
	"int32", "uint32",
	"int64", "uint64",
	"float", "double",
	"complex_int8", "complex_uint8",
	"complex_int16", "complex_uint16",
	"complex_int32", "complex_uint32",
	"complex_int64", "complex_uint64",
	"complex_float", "complex_double",
};


void wekuaMatrixRealPrintChar(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	int8_t real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 5 || y >= row-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= col-4) && (y < 4 || y >= row-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%5i", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= row-4)) || (y == 4 && (x < 4 || x >= col-4))){
				printf("%6s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 5 || y >= row-4)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintUchar(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	uint8_t real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 5 || y >= row-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= col-4) && (y < 4 || y >= row-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%5u", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= row-4)) || (y == 4 && (x < 4 || x >= col-4))){
				printf("%6s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 5 || y >= row-4)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintShort(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	int16_t real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 5 || y >= row-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= col-4) && (y < 4 || y >= row-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%7i", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= row-4)) || (y == 4 && (x < 4 || x >= col-4))){
				printf("%8s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 5 || y >= row-4)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintUshort(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	uint16_t real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 5 || y >= row-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= col-4) && (y < 4 || y >= row-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%7u", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= row-4)) || (y == 4 && (x < 4 || x >= col-4))){
				printf("%8s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 5 || y >= row-4)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintInt(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	int32_t real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 5 || y >= row-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= col-4) && (y < 4 || y >= row-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%12i", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= row-4)) || (y == 4 && (x < 4 || x >= col-4))){
				printf("%13s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 5 || y >= row-4)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintUint(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	uint32_t real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 5 || y >= row-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= col-4) && (y < 4 || y >= row-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%12u", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= row-4)) || (y == 4 && (x < 4 || x >= col-4))){
				printf("%13s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 5 || y >= row-4)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintLong(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	int64_t real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%22li", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 2 || x >= col-2))){
				printf("%23s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintUlong(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	uint64_t real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%22lu", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 2 || x >= col-2))){
				printf("%23s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintFloat(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	float real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 5 || y >= row-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= col-4) && (y < 4 || y >= row-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%14.5e", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= row-4)) || (y == 4 && (x < 4 || x >= col-4))){
				printf("%15s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 5 || y >= row-4)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixRealPrintDouble(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	double real;
	register uint64_t col, row;
	row = a->shape[0];
	col = a->shape[1];

	for (uint64_t y=0; y<row; y++){
		for (uint64_t x=0; x<col; x++){
			if (x == 0 && (y < 5 || y >= row-4)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 4 || x >= col-4) && (y < 4 || y >= row-4)){
				wekuaGetValueFromMatrix(a, y, x, &real, NULL, 0, NULL);
				printf("%14.5e", real);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 4 && (y < 5 || y >= row-4)) || (y == 4 && (x < 4 || x >= col-4))){
				printf("%15s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 5 || y >= row-4)){
				printf("\n");
			}
		}
	}
}



void wekuaMatrixComplexPrintChar(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[10];

	int8_t real, imag;

	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 10);
				wekuaGetValueFromMatrix(a, y, x, &real, &imag, 0, NULL);

				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%d%+dj", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%de", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%dj", imag);
				}else{
					sprintf(num, "%d", real);
				}
				printf("%10s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%11s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintUchar(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[10];

	uint8_t real, imag;

	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 10);
				wekuaGetValueFromMatrix(a, y, x, &real, &imag, 0, NULL);

				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%u+%uj", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%ue", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%uj", imag);
				}else{
					sprintf(num, "%u", real);
				}
				printf("%10s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%11s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintShort(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[14];

	int16_t real, imag;

	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 14);
				wekuaGetValueFromMatrix(a, y, x, &real, &imag, 0, NULL);

				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%d%+dj", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%de", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%dj", imag);
				}else{
					sprintf(num, "%d", real);
				}
				printf("%14s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%15s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintUshort(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[14];

	uint16_t real, imag;

	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 14);
				wekuaGetValueFromMatrix(a, y, x, &real, &imag, 0, NULL);

				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%u+%uj", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%ue", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%uj", imag);
				}else{
					sprintf(num, "%u", real);
				}
				printf("%14s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%15s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintInt(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[23];

	double real, imag;
	int32_t r, i;

	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 23);
				wekuaGetValueFromMatrix(a, y, x, &r, &i, 0, NULL);
				real = 1.0*r; imag = 1.0*i;

				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%.2e%+.2ej", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%.5e", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%.5ej", imag);
				}else{
					sprintf(num, "%.5e", real);
				}
				printf("%24s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%25s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintUint(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[23];

	double real, imag;
	uint32_t r, i;

	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 23);
				wekuaGetValueFromMatrix(a, y, x, &r, &i, 0, NULL);
				real = 1.0*r; imag = 1.0*i;

				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%.2e%+.2ej", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%.5e", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%.5ej", imag);
				}else{
					sprintf(num, "%.5e", real);
				}
				printf("%24s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%25s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintLong(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[23];

	double real, imag;
	int64_t r, i;

	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 23);
				wekuaGetValueFromMatrix(a, y, x, &r, &i, 0, NULL);
				real = 1.0*r; imag = 1.0*i;

				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%.2e%+.2ej", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%.5e", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%.5ej", imag);
				}else{
					sprintf(num, "%.5e", real);
				}
				printf("%24s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%25s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintUlong(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[23];

	double real, imag;
	uint64_t r, i;

	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 23);
				wekuaGetValueFromMatrix(a, y, x, &r, &i, 0, NULL);
				real = 1.0*r; imag = 1.0*i;

				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%.2e%+.2ej", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%.5e", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%.5ej", imag);
				}else{
					sprintf(num, "%.5e", real);
				}
				printf("%24s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%25s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintFloat(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[23];

	float real, imag;
	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];

	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 23);
				wekuaGetValueFromMatrix(a, y, x, &real, &imag, 0, NULL);
				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%.2e%+.2ej", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%.5e", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%.5ej", imag);
				}else{
					sprintf(num, "%.5e", real);
				}
				printf("%24s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%25s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}

void wekuaMatrixComplexPrintDouble(wmatrix a){
	if (a == NULL){
		return;
	}
	uint8_t d = 0;
	char num[23];
	double real, imag;
	uint32_t col, row;

	col = a->shape[1];
	row = a->shape[0];


	for (uint32_t y=0; y<row; y++){
		for (uint32_t x=0; x<col; x++){
			if (x == 0 && (y < 3 || y >= row-2)){
				if (d){
					printf("         ");
				}else{
					d ^= 1;
				}
			}
			if ((x < 2 || x >= col-2) && (y < 2 || y >= row-2)){
				memset(num, 0, 23);
				wekuaGetValueFromMatrix(a, y, x, &real, &imag, 0, NULL);
				if (real != 0.0 && imag != 0.0){
					sprintf(num, "%.2e%+.2ej", real, imag);
				}else if (real != 0.0 && imag == 0.0){
					sprintf(num, "%.5e", real);
				}else if (real == 0.0 && imag != 0.0){
					sprintf(num, "%.5ej", imag);
				}else{
					sprintf(num, "%.5e", real);
				}
				printf("%24s", num);
				if (y+1 != row || x+1 != col){
					printf(",");
				}
			}else if ((x == 2 && (y < 3 || y >= row-2)) || (y == 2 && (x < 3 || x >= col-2))){
				printf("%25s", "... ");
			}
			if (row > 1 && x == col-1 && (y < 3 || y >= row-2)){
				printf("\n");
			}
		}
	}
}


void (*wekuaMatrixComplexPrint[10])(wmatrix a) = {
	&wekuaMatrixComplexPrintChar, &wekuaMatrixComplexPrintUchar,
	&wekuaMatrixComplexPrintShort, &wekuaMatrixComplexPrintUshort,
	&wekuaMatrixComplexPrintInt, &wekuaMatrixComplexPrintUint,
	&wekuaMatrixComplexPrintLong, &wekuaMatrixComplexPrintUlong,
	&wekuaMatrixComplexPrintFloat, &wekuaMatrixComplexPrintDouble
};

void (*wekuaMatrixRealPrint[10])(wmatrix a) = {
	&wekuaMatrixRealPrintChar, &wekuaMatrixRealPrintUchar,
	&wekuaMatrixRealPrintShort, &wekuaMatrixRealPrintUshort,
	&wekuaMatrixRealPrintInt, &wekuaMatrixRealPrintUint,
	&wekuaMatrixRealPrintLong, &wekuaMatrixRealPrintUlong,
	&wekuaMatrixRealPrintFloat, &wekuaMatrixRealPrintDouble
};

void wekuaMatrixPrint(wmatrix a, uint32_t nw, cl_event *e){
	if (a == NULL){
		return;
	}
	clWaitForEvents(nw, e);
	printf("wmatrix([");
	if(a->com){
		wekuaMatrixComplexPrint[a->dtype](a);
	}else{
		wekuaMatrixRealPrint[a->dtype](a);
	}
	printf("], shape=(%ld, %ld), dtype=%s)\n", a->shape[0], a->shape[1], dtype_text[a->com*10+a->dtype]);
}
#include "wekua.h"
#include <math.h>

void runKernel(cl_command_queue cmd, cl_kernel kernel, uint32_t ndim, uint64_t *offsi, uint64_t *glosi, uint64_t *losi);

wmatrix *getUpperLowerBounds(wmatrix *a){
	wmatrix *degree, *b;
	double max;
	degree = wekuaMatrixResize(a, 1, a->c-1, 0.0, 0.0);

	b = wekuaAllocMatrix(a->ctx, 1, 2);
	wekuaMatrixAbs(degree);
	wekuaMatrixMax(degree, &max, NULL);

	b->raw_real[1] = 1.0 + max/fabs(a->raw_real[a->c-1]);
	
	wekuaFreeMatrix(degree);
	degree = wekuaCutMatrix(a, 1, a->c-1, 0, 1);
	wekuaMatrixAbs(degree);
	wekuaMatrixMax(degree, &max, NULL);

	b->raw_real[0] = fabs(a->raw_real[0])/(fabs(a->raw_real[0])+max);

	wekuaFreeMatrix(degree);

	return b;
}

wmatrix *getRoots(wmatrix *ran, uint32_t degree){
	wmatrix *roots, *radius, *angle;
	radius = wekuaMatrixRandUniform(ran->ctx, 1, degree, ran->raw_real[0], 0.0, ran->raw_real[1], 0.0, 0);
	angle = wekuaMatrixRandUniform(ran->ctx, 1, degree, 0.0, 0.0, CL_M_PI*2, 0.0, 0);
	roots = wekuaAllocComplexMatrix(ran->ctx, 1, degree);

	for (uint32_t x=0; x<degree; x++){
		roots->raw_real[x] = radius->raw_real[x]*cos(angle->raw_real[x]);
		roots->raw_imag[x] = radius->raw_real[x]*sin(angle->raw_real[x]);
	}
	wekuaFreeMatrix(radius);
	wekuaFreeMatrix(angle);
	return roots;
}

void complex_mul(double *a, double *b, double c, double d){
	double e, f;
	e = a[0]*c - b[0]*d;
	f = a[0]*d + b[0]*c;
	a[0] = e;
	b[0] = f;
}

void calc_poly(double r, double i, double *a, double *b, wmatrix *poly){
	a[0] = 0.0;
	b[0] = 0.0;
	double c = 1.0, d = 0.0;
	for (unsigned int x=0; x<poly->c; x++){
		a[0] += poly->raw_real[x]*c - poly->raw_imag[x]*d;
		b[0] += poly->raw_real[x]*d + poly->raw_imag[x]*c;
		complex_mul(&c, &d, r, i);
	}
}

wmatrix *calc_dev(wmatrix *poly){
	wmatrix *a = wekuaAllocComplexMatrix(poly->ctx, poly->r, poly->c-1);
	for (uint32_t x=1; x < poly->c; x++){
		a->raw_real[x-1] = x*poly->raw_real[x];
		a->raw_imag[x-1] = x*poly->raw_imag[x];
	}
	return a;
}

void calc_inv_complex(double *a, double *b){
	double c, d;
	c = a[0]/(a[0]*a[0]+b[0]*b[0]);
	d = -1.0*b[0]/(a[0]*a[0]+b[0]*b[0]);
	a[0] = c;
	b[0] = d;
}

void calc_ratio(double r, double i, double *fr, double *fi, wmatrix *poly, wmatrix *dpoly){
	double dr, di;

	calc_poly(r, i, fr, fi, poly);
	calc_poly(r, i, &dr, &di, dpoly);

	calc_inv_complex(&dr, &di);

	complex_mul(fr, fi, dr, di);
}


wmatrix *wekuaMatrixRoot(wmatrix *a){
	if (a == NULL){
		return NULL;
	}
	wmatrix *ran, *roots, *d;
	ran = getUpperLowerBounds(a);
	roots = getRoots(ran, a->c-1);

	if (a->com == 0){
		if (createComplexMatrix(a)){
			wekuaFreeMatrix(roots);
			return NULL;
		}
	}
	d = calc_dev(a);

	double ratior, ratioi, devr, devi;
	double ro, io;
	uint32_t valid;

	while (1){
		valid = 0;
		for (uint32_t i=0; i < roots->c; i++){
			calc_ratio(roots->raw_real[i], roots->raw_imag[i], &ratior, &ratioi, a, d);

			devr = 0.0; devi = 0.0;
			for (uint32_t x=0; x < roots->c; x++){
				if (x != i){
					ro = roots->raw_real[i]; io = roots->raw_imag[i];
					ro -= roots->raw_real[x]; io -= roots->raw_imag[x];

					calc_inv_complex(&ro, &io);
					devr += ro; devi += io;
				}
			}

			complex_mul(&devr, &devi, ratior, ratioi);
			devr = 1.0 - devr;
			devi *= -1.0;
			calc_inv_complex(&devr, &devi);
			complex_mul(&ratior, &ratioi, devr, devi);

			roots->raw_real[i] -= ratior;
			roots->raw_imag[i] -= ratioi;

			if (sqrt(ratior*ratior + ratioi*ratioi) < 1e-14){
				valid++;
			}
		}
		if (valid == roots->c){
			break;
		}
	}

	return roots;
}
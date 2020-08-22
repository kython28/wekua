#include "wekua.h"
#include <unistd.h>

wmatrix *flinear(wmatrix *a){
	return wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
}
wmatrix *fsigmoid(wmatrix *a){
	wmatrix *b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
	wekuaMatrixSub(b, a);
	wekuaMatrixDot(b, a);
	return b;
}
wmatrix *ftanh(wmatrix *a){
	wmatrix *b, *c;
	b = wekuaFillMatrix(a->ctx, a->shape[0], a->shape[1], 1.0, 0.0);
	c = wekuaMatrixCopy(a);
	wekuaMatrixDot(c, a);
	wekuaMatrixSub(b, c);
	wekuaFreeMatrix(c);
	return b;
}

wmatrix *(*acti_func[3])(wmatrix *a) = {&flinear, &fsigmoid, &ftanh};

void wekuaGradientDescent(double lr, warch *a, wmatrix *output, wmatrix *ow, wloss *l, double *real, double *imag){
	if (a == NULL || output == NULL || l == NULL){
		return;
	}else if (a->pseq == 0){
		return;
	}
	wmatrix *error, *f, *t, *p, **s, **wei, **cache;
	s = a->s;
	wei = a->weight;
	cache = a->cache;
	
	l->func(ow, output, real, imag);
	error = l->get_dev(ow, output);
	uint32_t pseq = a->pseq;

	s[pseq-2] = acti_func[a->acti_func_id[pseq-2]](output);

	wekuaMatrixDot(s[pseq-2], error);
	wekuaFreeMatrix(error);

	if (pseq > 2){
		for (uint32_t x=3; 0 < pseq-x+1; x++){
			s[pseq-x] = acti_func[a->acti_func_id[pseq-x]](cache[pseq-x+1]);
			p = wekuaMatrixResize(wei[pseq-x+1], wei[pseq-x+1]->shape[0]-1, wei[pseq-x+1]->shape[1], 0.0, 0.0);
			f = wekuaMatrixTrans(p);
			t = wekuaMatrixProduct(s[pseq-x+1], f);
			wekuaFreeMatrix(f);
			wekuaFreeMatrix(p);
			wekuaMatrixDot(s[pseq-x], t);
			wekuaFreeMatrix(t);
		}
	}

	for (uint32_t x=0; x<pseq-1; x++){
		p = wekuaMatrixResize(cache[x], cache[x]->shape[0], cache[x]->shape[1]+1, 1.0, 0.0);
		t = wekuaMatrixTrans(p);
		wekuaFreeMatrix(p);

		f = wekuaMatrixProduct(t, s[x]);
		wekuaFreeMatrix(t);
		wekuaMatrixDotScalar(f, lr, 0.0);

		wekuaMatrixSub(wei[x], f);
		wekuaFreeMatrix(f);
	}

	for (uint32_t x=0; x < pseq-1; x++){
		wekuaFreeMatrix(s[x]);
	}
}
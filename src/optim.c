#include "wekua.h"

void wekuaGradientDescent(double lr, double lri, warch *a, wmatrix *output, wmatrix *ow, wloss *l,
	double *real, double *imag){
	if (a == NULL || output == NULL || l == NULL){
		return;
	}else if (a->pseq == 0){
		return;
	}
	wmatrix *error, *f, *t, *p, **s, **wei, **cache;
	wacti **actis = a->acti_funcs;

	uint32_t pseq; int64_t *w_id;
	s = a->s;
	wei = a->weight;
	cache = a->cache;
	w_id = a->w_id;

	l->func(ow, output, real, imag);
	error = l->get_dev(ow, output);
	pseq = a->pseq;

	s[pseq-2] = getDevWekuaActi(actis[w_id[pseq-2]], output);

	wekuaMatrixDot(s[pseq-2], error);
	wekuaFreeMatrix(error);

	if (pseq > 2){
		for (uint32_t x=3; 0 < pseq-x+1; x++){
			s[pseq-x] =  getDevWekuaActi(actis[w_id[pseq-x]], cache[pseq-x+1]);
			p = wekuaCutMatrix(wei[w_id[pseq-x+1]], 0, wei[w_id[pseq-x+1]]->shape[1], 0, wei[w_id[pseq-x+1]]->shape[0]-1);
			f = wekuaMatrixTrans(p);
			t = wekuaMatrixProduct(s[pseq-x+1], f);
			wekuaMatrixDot(s[pseq-x], t);
			wekuaFreeMatrix(f);
			wekuaFreeMatrix(p);
			wekuaFreeMatrix(t);
		}
	}

	for (uint32_t x=0; x < pseq-1; x++){
		p = wekuaMatrixResize(cache[x], cache[x]->shape[0], cache[x]->shape[1]+1, 1.0, 0.0);
		t = wekuaMatrixTrans(p);
		wekuaFreeMatrix(p);

		f = wekuaMatrixProduct(t, s[x]);
		wekuaFreeMatrix(t);
		wekuaMatrixDotScalar(f, lr, lri);

		wekuaMatrixSub(wei[w_id[x]], f);
		wekuaFreeMatrix(f);
	}
	for (uint32_t x=0; x < pseq-1; x++){
		wekuaFreeMatrix(s[x]);
		s[x] = NULL;
	}
}
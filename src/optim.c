#include "wekua.h"

void Backward(wloss *l, wmatrix *output, wmatrix *ow, wmatrix **s, wmatrix **wei, wmatrix **cache,
	wacti **actis, int64_t *w_id, uint32_t pseq, uint32_t nw, cl_event *be){
	wmatrix *error, *f, *t, *p;
	wekuaContext *ctx = output->ctx;
	cl_event e[2];

	error = l->get_dev(ow, output, nw, be);

	s[pseq-2] = getDevWekuaActi(actis[w_id[pseq-2]], output, 0, NULL);

	wekuaMatrixDot(s[pseq-2], error, 0, NULL, e);
	wekuaFreeMatrix(error, 1, e);

	clReleaseEvent(e[0]);

	if (pseq > 2){
		for (uint32_t x=3; 0 < pseq-x+1; x++){
			p = cache[pseq-x+1];
			f = s[pseq-x+1];
			t = wekuaCutMatrix(p, 0, p->shape[1]-1, 0, p->shape[0]);

			s[pseq-x] = getDevWekuaActi(actis[w_id[pseq-x]], t, 0, NULL);
			wekuaFreeMatrix(t, 0, NULL);

			t = wei[w_id[pseq-x+1]];
			p = wekuaCutMatrix(t, 0, t->shape[1], 0, t->shape[0]-1);
			t = wekuaFillMatrix(ctx, f->shape[0], p->shape[0], 0.0, 0.0);

			wekuaBlasGemm(1.0, 0.0, 0, f, 1, p, 0.0, 0.0, t, 0, NULL, e);

			wekuaMatrixDot(s[pseq-x], t, 1, e, &e[1]);
			wekuaFreeMatrix(p, 1, &e[1]);
			wekuaFreeMatrix(t, 0, NULL);

			clReleaseEvent(e[0]);
			clReleaseEvent(e[1]);
		}
	}
}

void runWekuaOptim(woptim *optim, wmatrix *output, wmatrix *ow, wloss *l, uint32_t nw, cl_event *be){
	optim->step(optim->data, optim->arch, output, ow, l, nw, be);
}

void wGD(void **data, warch *a, wmatrix *output, wmatrix *ow, wloss *l, uint32_t nw, cl_event *be){
	if (a == NULL || output == NULL || l == NULL){
		return;
	}else if (a->pseq == 0){
		return;
	}
	cl_event *e, *befo = NULL;

	double lr, lri;
	lr = ((double*)data[0])[0];
	lri = ((double*)data[0])[1];

	wmatrix *t, **s, **wei, **cache;
	wacti **actis = a->acti_funcs;

	uint32_t pseq, we=0; int64_t *w_id;
	s = a->s;
	wei = a->weight;
	cache = a->cache;
	w_id = a->w_id;
	pseq = a->pseq;

	Backward(l, output, ow, s, wei, cache, actis, w_id, pseq, nw, be);

	pseq--;
	e = (cl_event*) calloc(pseq, sizeof(cl_event));


	for (uint32_t x=0; x < pseq; x++){
		t = cache[x];

		wekuaBlasGemm(lr, lri, 1, t, 0, s[x], 1.0, 0.0, wei[w_id[x]], we, befo, &e[x]);
		befo = &e[x];
		if (we == 0){
			we++;
		}
	}
	clWaitForEvents(1, befo);
	for (uint32_t x=0; x < pseq; x++){
		clReleaseEvent(e[x]);
	}
	free(e);

	for (uint32_t x=0; x < pseq; x++){
		wekuaFreeMatrix(s[x], 0, NULL);
		s[x] = NULL;
	}
}

woptim *wekuaGradientDescent(double lr, double lri, warch *a){
	woptim *optim = calloc(1, sizeof(woptim));
	optim->step = &wGD;
	optim->arch = a;
	optim->data = calloc(1, sizeof(void**));
	optim->data[0] = calloc(2, sizeof(double*));
	((double*)optim->data[0])[0] = -1.0*lr;
	((double*)optim->data[0])[1] = -1.0*lri;
	return optim;
}

void wekuaFreeOptimGD(woptim *opti, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (opti == NULL){
		return;
	}
	free(opti->data[0]);
	free(opti->data);
	free(opti);
}

void wGDM(void **data, warch *a, wmatrix *output, wmatrix *ow, wloss *l, uint32_t nw, cl_event *be){
	if (a == NULL || output == NULL || l == NULL){
		return;
	}else if (a->pseq == 0){
		return;
	}
	double lr, lri, momentum, imomentum;
	lr = ((double*)data[0])[0];
	lri = ((double*)data[0])[1];
	momentum = ((double*)data[0])[2];
	imomentum = ((double*)data[0])[3];
	cl_event e[2];

	wmatrix *t, *p, **s, **wei, **cache, **momen;
	wacti **actis = a->acti_funcs;

	uint32_t pseq; int64_t *w_id;
	s = a->s;
	wei = a->weight;
	cache = a->cache;
	w_id = a->w_id;
	pseq = a->pseq;
	momen = (wmatrix**)data[1];

	Backward(l, output, ow, s, wei, cache, actis, w_id, pseq, nw, be);

	pseq--;

	for (uint32_t x=0; x < pseq; x++){
		t = cache[x];

		p = momen[w_id[x]];

		wekuaBlasGemm(1.0-momentum, -1.0*imomentum, 1, t, 0, s[x], momentum, imomentum, p, 0, NULL, e);

		wekuaBlasAxpy(lr, lri, p, wei[w_id[x]], 1, e, &e[1]);
		for (uint32_t j=0; j<2; j++) clReleaseEvent(e[j]);
	}
	for (uint32_t x=0; x < pseq; x++){
		wekuaFreeMatrix(s[x], 0, NULL);
		s[x] = NULL;
	}
}

woptim *wekuaGradientDescentMomentum(double lr, double lri, double momentum, double imomentum, warch *a){
	woptim *optim = calloc(1, sizeof(woptim));
	optim->step = &wGDM;
	optim->arch = a;
	optim->data = calloc(2, sizeof(void*));

	optim->data[0] = calloc(4, sizeof(double));
	((double*)optim->data[0])[0] = -1.0*lr;
	((double*)optim->data[0])[1] = -1.0*lri;
	((double*)optim->data[0])[2] = momentum;
	((double*)optim->data[0])[3] = imomentum;


	optim->data[1] = (wmatrix**)calloc(a->nmodule[2], sizeof(wmatrix*));
	wmatrix *w;
	for (uint32_t x=0; x<a->nmodule[2]; x++){
		w = a->weight[x];
		((wmatrix**)optim->data[1])[x] = wekuaFillMatrix(w->ctx, w->shape[0], w->shape[1], 0.0, 0.0);
	}
	return optim;
}

void wekuaFreeOptimGDM(woptim *opti, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (opti == NULL){
		return;
	}
	free(opti->data[0]);
	wmatrix **mo = (wmatrix**)opti->data[1];
	for (uint32_t x=0; x<opti->arch->nmodule[2]; x++){
		wekuaFreeMatrix(mo[x], 0, NULL);
	}
	free(opti->data[1]);
	free(opti);
}

void wAdaGrad(void **data, warch *a, wmatrix *output, wmatrix *ow, wloss *l, uint32_t nw, cl_event *be){
	if (a == NULL || output == NULL || l == NULL){
		return;
	}else if (a->pseq == 0){
		return;
	}
	double lr, lri;
	lr = ((double*)data[0])[0];
	lri = ((double*)data[0])[1];
	cl_event e[5];

	wmatrix *f, *t, *p, **s, **wei, **cache, **grad, **epli;
	wacti **actis = a->acti_funcs;
	wekuaContext *ctx = output->ctx;

	uint32_t pseq; int64_t *w_id;
	s = a->s;
	wei = a->weight;
	cache = a->cache;
	w_id = a->w_id;
	pseq = a->pseq;
	grad = (wmatrix**)data[1];
	epli = (wmatrix**)data[2];

	Backward(l, output, ow, s, wei, cache, actis, w_id, pseq, nw, be);

	pseq--;

	for (uint32_t x=0; x < pseq; x++){
		t = cache[x]; p = s[x];
		f = wekuaFillMatrix(ctx, t->shape[1], p->shape[1], 0.0, 0.0);
		wekuaBlasGemm(1.0, 0.0, 1, t, 0, p, 0.0, 0.0, f, 0, NULL, e);

		t = wekuaMatrixCopy(f, 1, e, &e[1]);
		wekuaMatrixDot(t, t, 1, &e[1], &e[2]);

		p = grad[w_id[x]];

		wekuaMatrixAdd(p, t, 1, &e[2], &e[3]);
		wekuaFreeMatrix(t, 1, &e[3]);
		for (uint32_t j=0; j<4; j++){
			clReleaseEvent(e[j]);
		}

		t = wekuaMatrixCopy(p, 0, NULL, e);
		wekuaMatrixAdd(t, epli[w_id[x]], 1, e, &e[1]);
		wekuaMatrixPowr(t, -0.5, 0.0, 1, &e[1], &e[2]);
		wekuaMatrixDot(t, f, 1, &e[2], &e[3]);
		wekuaBlasAxpy(lr, lri, t, wei[w_id[x]], 1, &e[3], &e[4]);

		wekuaFreeMatrix(t, 1, &e[4]);
		wekuaFreeMatrix(f, 0, NULL);

		for (uint32_t j=0; j<5; j++){
			clReleaseEvent(e[j]);
		}
	}
	for (uint32_t x=0; x < pseq; x++){
		wekuaFreeMatrix(s[x], 0, NULL);
		s[x] = NULL;
	}
}

woptim *wekuaAdaGrad(double lr, double lri, warch *a){
	woptim *optim = calloc(1, sizeof(woptim));
	optim->step = &wAdaGrad;
	optim->arch = a;
	optim->data = calloc(3, sizeof(void*));

	optim->data[0] = calloc(2, sizeof(double));
	((double*)optim->data[0])[0] = -1.0*lr;
	((double*)optim->data[0])[1] = -1.0*lri;

	optim->data[1] = (wmatrix**) calloc(a->nmodule[2], sizeof(wmatrix*));
	optim->data[2] = (wmatrix**) calloc(a->nmodule[2], sizeof(wmatrix*));
	wmatrix *w;
	for (uint32_t x=0; x<a->nmodule[2]; x++){
		w = a->weight[x];
		((wmatrix**)optim->data[1])[x] = wekuaFillMatrix(w->ctx, w->shape[0], w->shape[1], 0.0, 0.0);
		((wmatrix**)optim->data[2])[x] = wekuaFillMatrix(w->ctx, w->shape[0], w->shape[1], CL_DBL_EPSILON, 0.0);
	}

	return optim;
}

void wekuaFreeOptimAdaGrad(woptim *opti, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (opti == NULL){
		return;
	}
	free(opti->data[0]);
	wmatrix **gr = (wmatrix**)opti->data[1];
	wmatrix **ep = (wmatrix**)opti->data[2];
	for (uint32_t x=0; x<opti->arch->nmodule[2]; x++){
		wekuaFreeMatrix(gr[x], 0, NULL);
		wekuaFreeMatrix(ep[x], 0, NULL);
	}
	free(opti->data[1]);
	free(opti);
}

void wRMS(void **data, warch *a, wmatrix *output, wmatrix *ow, wloss *l, uint32_t nw, cl_event *be){
	if (a == NULL || output == NULL || l == NULL){
		return;
	}else if (a->pseq == 0){
		return;
	}
	double lr, lri, beta, ibeta;
	lr = ((double*)data[0])[0];
	lri = ((double*)data[0])[1];
	beta = ((double*)data[0])[2];
	ibeta = ((double*)data[0])[3];
	cl_event e[5];

	wmatrix *f, *t, *p, **s, **wei, **cache, **grad, **epli;
	wacti **actis = a->acti_funcs;
	wekuaContext *ctx = output->ctx;

	uint32_t pseq; int64_t *w_id;
	s = a->s;
	wei = a->weight;
	cache = a->cache;
	w_id = a->w_id;
	pseq = a->pseq;
	grad = (wmatrix**)data[1];
	epli = (wmatrix**)data[2];

	Backward(l, output, ow, s, wei, cache, actis, w_id, pseq, nw, be);

	pseq--;

	for (uint32_t x=0; x < pseq; x++){
		t = cache[x]; p = s[x];
		f = wekuaFillMatrix(ctx, t->shape[1], p->shape[1], 0.0, 0.0);
		wekuaBlasGemm(1.0, 0.0, 1, t, 0, p, 0.0, 0.0, f, 0, NULL, e);

		t = wekuaMatrixCopy(f, 1, e, &e[1]);
		wekuaMatrixDot(t, t, 1, &e[1], &e[2]);

		p = grad[w_id[x]];

		wekuaMatrixDotScalar(p, beta, ibeta, 1, &e[2], &e[3]);
		wekuaBlasAxpy(1.0-beta, -1.0*ibeta, t, p, 1, &e[3], &e[4]);
		wekuaFreeMatrix(t, 1, &e[4]);
		for (uint32_t j=0; j<5; j++){
			clReleaseEvent(e[j]);
		}

		t = wekuaMatrixCopy(p, 0, NULL, e);
		wekuaMatrixAdd(t, epli[w_id[x]], 1, e, &e[1]);
		wekuaMatrixPowr(t, -0.5, 0.0, 1, &e[1], &e[2]);
		wekuaMatrixDot(t, f, 1, &e[2], &e[3]);
		wekuaBlasAxpy(lr, lri, t, wei[w_id[x]], 1, &e[3], &e[4]);

		wekuaFreeMatrix(t, 1, &e[4]);
		wekuaFreeMatrix(f, 0, NULL);

		for (uint32_t j=0; j<5; j++){
			clReleaseEvent(e[j]);
		}
	}
	for (uint32_t x=0; x < pseq; x++){
		wekuaFreeMatrix(s[x], 0, NULL);
		s[x] = NULL;
	}
}

woptim *wekuaRMSprop(double lr, double lri, double beta, double ibeta, warch *a){
	woptim *optim = calloc(1, sizeof(woptim));
	optim->step = &wRMS;
	optim->arch = a;
	optim->data = calloc(3, sizeof(void*));

	optim->data[0] = calloc(4, sizeof(double));
	((double*)optim->data[0])[0] = -1.0*lr;
	((double*)optim->data[0])[1] = -1.0*lri;
	((double*)optim->data[0])[2] = beta;
	((double*)optim->data[0])[3] = ibeta;

	optim->data[1] = (wmatrix**) calloc(a->nmodule[2], sizeof(wmatrix*));
	optim->data[2] = (wmatrix**) calloc(a->nmodule[2], sizeof(wmatrix*));
	wmatrix *w;
	for (uint32_t x=0; x<a->nmodule[2]; x++){
		w = a->weight[x];
		((wmatrix**)optim->data[1])[x] = wekuaFillMatrix(w->ctx, w->shape[0], w->shape[1], 0.0, 0.0);
		((wmatrix**)optim->data[2])[x] = wekuaFillMatrix(w->ctx, w->shape[0], w->shape[1], CL_DBL_EPSILON, 0.0);
	}

	return optim;
}

void wekuaFreeOptimRMSprop(woptim *opti, uint32_t nw, cl_event *be){
	clWaitForEvents(nw, be);
	if (opti == NULL){
		return;
	}
	free(opti->data[0]);
	wmatrix **gr = (wmatrix**)opti->data[1];
	wmatrix **ep = (wmatrix**)opti->data[2];
	for (uint32_t x=0; x<opti->arch->nmodule[2]; x++){
		wekuaFreeMatrix(gr[x], 0, NULL);
		wekuaFreeMatrix(ep[x], 0, NULL);
	}
	free(opti->data[1]);
	free(opti);
}
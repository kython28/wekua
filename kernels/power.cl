#include "/usr/lib/wekua_kernels/dtype.cl"

void complex_mul(wks *a, wks *b, wks c, wks d){
	wks e, f, g, h;
	g = a[0]; h = b[0];
	e = g*c - h*d;
	f = g*d + h*c;
	a[0] = e;
	b[0] = f;
}

void step_one(wks *a, wks *b, wks r, wks h, wks y){
	wks c = cosh(h) - sinh(h);
	a[0] = cos(y*r)*c;
	b[0] = sin(y*r)*c;
}

void step_two(wks *a, wks *b, wks h, wks r, wks x){
	wks er, co, si, mwo, awo;
	er = exp(r);
	co = cos(h)*er; si = sin(h)*er;

	awo = atan2(si, co);
	mwo = pow(sqrt(co*co + si*si), x);
	a[0] = mwo*cos(awo*x);
	b[0] = mwo*sin(awo*x);
}

__kernel void power(__global wks *a, __global wks *b,
	__global wks *c, __global wks *d,
	wks alpha, wks beta,
	unsigned long col, unsigned char om, unsigned char com){
	unsigned long i = get_global_id(0);
	unsigned long j = get_global_id(1);
	wks aa, bb, r, h, so, soi;

	unsigned long current = i*col+j;

	if (com){
		aa = a[current]; bb = b[current];
		r = 0.5*log(aa*aa + bb*bb);
		h = atan2(bb,aa);

		if (om){
			step_one(&so, &soi, r, h, d[current]);
			step_two(&aa, &bb, r, h, c[current]);
		}else{
			step_one(&so, &soi, r, h, beta);
			step_two(&aa, &bb, r, h, alpha);
		}

		complex_mul(&aa, &bb, so, soi);

		a[current] = aa;
		b[current] = bb;
	}else{
		if (om){
			a[current] = pow(a[current], c[current]);
		}else{
			a[current] = pow(a[current], alpha);
		}
	}
}
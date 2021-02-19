#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void adadelta(
	__global wk *gra1r, __global wk* gra1i,
	__global wk *gra2r, __global wk *gra2i,
	__global wk *err, __global wk *erri,
	__global wk *wr, __global wk *wi,

	wks alpha, wks alphai,
	wks beta, wks betai,

	unsigned long long col, unsigned char com
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wk t_gra1, t_gra1i, t_gra2, t_gra2i, error = err[i], errori;
	if (com){

	}else{
		t_gra1 = beta*gra1r[i];
		t_gra1 += (1 - beta)*error*error;

		t_gra2 = gra2r[i];

		#if dtype == 8
		error /= -sqrt(t_gra1 + FLT_EPSILON);
		wr[i] += error*sqrt(t_gra2 +FLT_EPSILON);
		#else
		error /= -sqrt(t_gra1 + DBL_EPSILON);
		wr[i] += error*sqrt(t_gra2 + DBL_EPSILON);
		#endif

		error *= alpha;
		error *= error;
		t_gra2 *= beta;
		t_gra2 += (1-beta)*error;
	}
	gra1r[i] = t_gra1;
	gra2r[i] = t_gra2;
}
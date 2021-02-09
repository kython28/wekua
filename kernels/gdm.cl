#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void gdm(
	__global wk *err, __global wk *erri,
	__global wk *wr, __global wk *wi,
	__global wk *vr, __global wk *vi,

	wks alphar, wks alphai,
	wks betar, wks betai,

	unsigned long col, unsigned char com
){
	unsigned long i = get_global_id(0)*col + get_global_id(1);

	wk t_vr = vr[i], t_vi, t_err = err[i], t_erri;
	if (com){
		t_vi = vi[i];
		t_erri = erri[i];

		complex_mul_scal(&t_vr, &t_vi, betar, betai);
		complex_mul_scal(&t_err, &t_erri, alphar, alphai);

		t_vr += t_err;
		t_vi += t_erri;

		wi[i] -= t_vi;
	}else{
		t_vr = t_vr*betar + alphar*t_err;
	}
	wr[i] -= t_vr;
	vr[i] = t_vr;
}
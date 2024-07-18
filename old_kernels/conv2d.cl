#include "/usr/lib/wekua_kernels/dtype.cl"

__kernel void conv2d(
	__global wk *kernelr, __global wk *kerneli,
	__global wks *biar, __global wks *biasi,
	__global wk *imgr, __global wk *imgi,
	__global wk *outputr, __global wk *outputi,
	unsigned long col_kernel, unsigned long row_kernel, unsigned long layers,
	unsigned long img_size, unsigned long col_img, unsigned long stride, unsigned long sample_num,
	unsigned long output_size, unsigned long output_col
){
	unsigned long x = get_global_id(0);
	unsigned long y = get_global_id(1)>>1;
	unsigned long z = get_global_id(2);

	unsigned long kernel_layer = z*layers, kernel_offset;
	unsigned long img_sample_offset, img_pixel_offset, output_offset;
	unsigned long img_offset_x, img_offset_y0, img_offset_y1;

	output_offset = y*output_col + x*layers + z;
	img_sample_offset = 0;

	img_offset_x = x*stride;
	img_offset_y0 = y*stride;
	img_offset_y1 = img_offset_y0 + 1;

	#if u_bias
	wk bias = biar[z];
	#endif

	wk r1, r2;
	wk g0, g1, g2;
	wk t1, t2;
	wk d0, d1, d2, d3, d4, d5;
	wk m0, m1, m2, m3;

	for (unsigned long n=0; n<sample_num; n++){
		#if u_bias
		r1 = bias; r2 = bias;
		#else
		r1 = 0; r2 = 0;
		#endif
		for (unsigned long j=0; j<col_kernel; j+=3){
			kernel_offset = kernel_layer + j;
			g0 = kernelr[kernel_offset];
			g2 = kernelr[kernel_offset + 1];
			g2 = kernelr[kernel_offset + 2];

			t1 = (g0 + g1 + g2)<<1;
			t2 = (g0 - g1 + g2)<<1;

			img_pixel_offset = img_sample_offset + img_offset_y0*col_img + j;
			d0 = imgr[img_pixel_offset];
			d1 = imgr[img_pixel_offset + 1];
			d2 = imgr[img_pixel_offset + 2];

			img_pixel_offset = img_sample_offset + img_offset_y1*col_img + j;

			d3 = imgr[img_pixel_offset];
			d4 = imgr[img_pixel_offset + 1];
			d5 = imgr[img_pixel_offset + 2];
			for (unsigned long i=0;;){
				m0 = (d0 - d2)*g0;
				m1 = (d1 - d3)*g2;
				m2 = (d1 + d2)*t1;
				m3 = (d2 - d1)*t2;

				r1 += m0 + m1 + m2;
				r2 += m1 - m2 - m3;

				i++;
				if (i==row_kernel) break;
				d0 = d3;
				d1 = d4;
				d2 = d5;
				img_pixel_offset = img_sample_offset + (img_offset_y1 + 1)*col_img + j;
				d3 = imgr[img_pixel_offset];
				d4 = imgr[img_pixel_offset + 1];
				d5 = imgr[img_pixel_offset + 2];
				kernel_offset = kernel_layer + i*col_kernel + j;
			}
		}
		outputr[output_offset] = r1;
		outputr[output_offset + output_col] = r2;

		output_offset += output_size;
		img_sample_offset += img_size;
	}
}
#ifndef com
#define com 0
#endif

#ifndef dtype
#define dtype 9
#endif

#ifndef wk_width
#define wk_width 1
#endif


#if dtype == 0

#define WK_INT_MAX CHAR_MAX
#define WK_UINT_MAX UCHAR_MAX

typedef uchar uwks;
typedef char wks;

#define WKS_IS_UNSIGNED 0

#if wk_width == 1
typedef char wk;
#elif wk_width == 2
typedef char2 wk;
#elif wk_width == 4
typedef char4 wk;
#elif wk_width == 8
typedef char8 wk;
#elif wk_width == 16
typedef char16 wk;
#endif

#define convert_T convert_char_sat_rte

#elif dtype == 1

#define WK_INT_MAX UCHAR_MAX
#define WK_UINT_MAX UCHAR_MAX

typedef uchar uwks;
typedef uchar wks;

#define WKS_IS_UNSIGNED 1

#if wk_width == 1
typedef uchar wk;
#elif wk_width == 2
typedef uchar2 wk;
#elif wk_width == 4
typedef uchar4 wk;
#elif wk_width == 8
typedef uchar8 wk;
#elif wk_width == 16
typedef uchar16 wk;
#endif

#define convert_T convert_uchar_sat_rte

#elif dtype == 2

#define WK_INT_MAX SHRT_MAX
#define WK_UINT_MAX USHRT_MAX

typedef ushort uwks;
typedef short wks;

#define WKS_IS_UNSIGNED 0

#if wk_width == 1
typedef short wk;
#elif wk_width == 2
typedef short2 wk;
#elif wk_width == 4
typedef short4 wk;
#elif wk_width == 8
typedef short8 wk;
#elif wk_width == 16
typedef short16 wk;
#endif

#define convert_T convert_short_sat_rte

#elif dtype == 3

#define WK_INT_MAX USHRT_MAX
#define WK_UINT_MAX USHRT_MAX

typedef ushort uwks;
typedef ushort wks;

#define WKS_IS_UNSIGNED 1

#if wk_width == 1
typedef ushort wk;
#elif wk_width == 2
typedef ushort2 wk;
#elif wk_width == 4
typedef ushort4 wk;
#elif wk_width == 8
typedef ushort8 wk;
#elif wk_width == 16
typedef ushort16 wk;
#endif

#define convert_T convert_ushort_sat_rte

#elif dtype == 4

#define WK_INT_MAX INT_MAX
#define WK_UINT_MAX UINT_MAX

typedef uint uwks;
typedef int wks;

#define WKS_IS_UNSIGNED 0

#if wk_width == 1
typedef int wk;
#elif wk_width == 2
typedef int2 wk;
#elif wk_width == 4
typedef int4 wk;
#elif wk_width == 8
typedef int8 wk;
#elif wk_width == 16
typedef int16 wk;
#endif

#define convert_T convert_int_sat_rte

#elif dtype == 5

#define WK_INT_MAX UINT_MAX
#define WK_UINT_MAX UINT_MAX

typedef uint uwks;
typedef uint wks;

#define WKS_IS_UNSIGNED 1

#if wk_width == 1
typedef uint wk;
#elif wk_width == 2
typedef uint2 wk;
#elif wk_width == 4
typedef uint4 wk;
#elif wk_width == 8
typedef uint8 wk;
#elif wk_width == 16
typedef uint16 wk;
#endif

#define convert_T convert_uint_sat_rte

#elif dtype == 6

#define WK_INT_MAX LONG_MAX
#define WK_UINT_MAX ULONG_MAX

typedef ulong uwks;
typedef long wks;

#define WKS_IS_UNSIGNED 1

#if wk_width == 1
typedef long wk;
#elif wk_width == 2
typedef long2 wk;
#elif wk_width == 4
typedef long4 wk;
#elif wk_width == 8
typedef long8 wk;
#elif wk_width == 16
typedef long16 wk;
#endif

#define convert_T convert_long_sat_rte

#elif dtype == 7

#define WK_INT_MAX ULONG_MAX
#define WK_UINT_MAX ULONG_MAX

typedef ulong uwks;
typedef ulong wks;

#define WKS_IS_UNSIGNED 1

#if wk_width == 1
typedef ulong wk;
#elif wk_width == 2
typedef ulong2 wk;
#elif wk_width == 4
typedef ulong4 wk;
#elif wk_width == 8
typedef ulong8 wk;
#elif wk_width == 16
typedef ulong16 wk;
#endif

#define convert_T convert_ulong_sat_rte

#elif dtype == 8

typedef float wks;

#if wk_width == 1
typedef float wk;
#elif wk_width == 2
typedef float2 wk;
#elif wk_width == 4
typedef float4 wk;
#elif wk_width == 8
typedef float8 wk;
#elif wk_width == 16
typedef float16 wk;
#endif

#define convert_T convert_float

#elif dtype == 9

typedef double wks;

#if wk_width == 1
typedef double wk;
#elif wk_width == 2
typedef double2 wk;
#elif wk_width == 4
typedef double4 wk;
#elif wk_width == 8
typedef double8 wk;
#elif wk_width == 16
typedef double16 wk;
#endif

#define convert_T convert_double

#endif

#if wk_width > 1

#if dtype < 9 // PoCL doesn't support _cl_dot with float

wks __attribute__((overloadable)) dot(wk aa, wk bb){
	wk cc = aa*bb;
	wks r = cc[0];

	#if wk_width >= 2
	r += cc[1];
	#endif

	#if wk_width >= 4
	r += cc[2];
	r += cc[3];
	#endif

	#if wk_width >= 8
	r += cc[4];
	r += cc[5];
	r += cc[6];
	r += cc[7];
	#endif

	#if wk_width == 16
	r += cc[8];
	r += cc[9];
	r += cc[10];
	r += cc[11];
	r += cc[12];
	r += cc[13];
	r += cc[14];
	r += cc[15];
	#endif

	return r;
}

#endif

wks sum(wk a){
	wks r = a[0];

	#if wk_width >= 2
	r += a[1];
	#endif

	#if wk_width >= 4
	r += a[2];
	r += a[3];
	#endif

	#if wk_width >= 8
	r += a[4];
	r += a[5];
	r += a[6];
	r += a[7];
	#endif

	#if wk_width == 16
	r += a[8];
	r += a[9];
	r += a[10];
	r += a[11];
	r += a[12];
	r += a[13];
	r += a[14];
	r += a[15];
	#endif

	return r;
}

#endif

void complex_mul_vector_vector(wk *a, wk *b, wk c, wk d){
	wk k1, k2, k3, aa, bb;

	aa = a[0]; bb = b[0];

	k1 = c*(aa + bb);
	k2 = aa*(d - c);
	k3 = bb*(c + d);

	a[0] = k1 - k3;
	b[0] = k1 + k2;
}

#if dtype >= 8
void complex_mul_vector_scalar(wk *a, wk *b, wks c, wks d){
	wk k1, k2, k3, aa, bb;

	aa = a[0]; bb = b[0];

	k1 = c*(aa + bb);
	k2 = aa*(d - c);
	k3 = bb*(c + d);

	a[0] = k1 - k3;
	b[0] = k1 + k2;
}
#endif

#define COMPLEX_MUL_K(T) \
	T k1, k2, k3;

#define COMPLEX_MUL(a, b, c, d) \
	k1 = c*(a + b); \
	k2 = a*(d - c); \
	k3 = b*(c + d); \
	a = k1 - k3; \
	b = k1 + k2; \

void calc_inv_complex(wk *a, wk *b){
	wk c, d, aa, bb;
	aa = a[0]; bb = b[0];

	c = aa;
	d = -bb;

	aa = (aa*aa + bb*bb);

	c /= aa;
	d /= aa;
	
	a[0] = c;
	b[0] = d;
}

void calc_inv_complex_scalar(wks *a, wks *b){
	wks c, d, aa, bb;
	aa = a[0]; bb = b[0];

	c = aa;
	d = -bb;

	aa = (aa*aa + bb*bb);

	c /= aa;
	d /= aa;
	
	a[0] = c;
	b[0] = d;
}


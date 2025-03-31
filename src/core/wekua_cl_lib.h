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

#if wk_width >= 2
typedef char2 wk2;
#endif

#if wk_width >= 4
typedef char4 wk4;
#endif

#if wk_width >= 8
typedef char8 wk8;
#endif

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

#if wk_width >= 2
typedef uchar2 wk2;
#endif

#if wk_width >= 4
typedef uchar4 wk4;
#endif

#if wk_width >= 8
typedef uchar8 wk8;
#endif

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

#if wk_width >= 2
typedef short2 wk2;
#endif

#if wk_width >= 4
typedef short4 wk4;
#endif

#if wk_width >= 8
typedef short8 wk8;
#endif

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

#if wk_width >= 2
typedef ushort2 wk2;
#endif

#if wk_width >= 4
typedef ushort4 wk4;
#endif

#if wk_width >= 8
typedef ushort8 wk8;
#endif

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

#if wk_width >= 2
typedef int2 wk2;
#endif

#if wk_width >= 4
typedef int4 wk4;
#endif

#if wk_width >= 8
typedef int8 wk8;
#endif

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

#if wk_width >= 2
typedef uint2 wk2;
#endif

#if wk_width >= 4
typedef uint4 wk4;
#endif

#if wk_width >= 8
typedef uint8 wk8;
#endif

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

#if wk_width >= 2
typedef long2 wk2;
#endif

#if wk_width >= 4
typedef long4 wk4;
#endif

#if wk_width >= 8
typedef long8 wk8;
#endif

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

#if wk_width >= 2
typedef ulong2 wk2;
#endif

#if wk_width >= 4
typedef ulong4 wk4;
#endif

#if wk_width >= 8
typedef ulong8 wk8;
#endif

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

#if wk_width >= 2
typedef float2 wk2;
#endif

#if wk_width >= 4
typedef float4 wk4;
#endif

#if wk_width >= 8
typedef float8 wk8;
#endif

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

#if wk_width >= 2
typedef double2 wk2;
#endif

#if wk_width >= 4
typedef double4 wk4;
#endif

#if wk_width >= 8
typedef double8 wk8;
#endif

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

inline wks sum(wk a) {
#if wk_width == 1
	return a;
#elif wk_width == 2
	return a.lo + a.hi;
#elif wk_width == 4
    const wk2 temp = a.lo + a.hi;
    return temp.lo + temp.hi;
#elif wk_width == 8
	const wk4 temp = a.lo + a.hi;
    const wk2 temp2 = temp.lo + temp.hi;
    return temp2.lo + temp2.hi;
#elif wk_width == 16
	const wk8 temp = a.lo + a.hi;
	const wk4 temp2 = temp.lo + temp.hi;
    const wk2 temp3 = temp2.lo + temp2.hi;
    return temp3.lo + temp3.hi;
#endif
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

#define COMPLEX_S_MUL_K(T) \
	T k1_s, k2_s, k3_s;

#define COMPLEX_S_MUL(a, b, c, d) \
	k1_s = c*(a + b); \
	k2_s = a*(d - c); \
	k3_s = b*(c + d); \
	a = k1_s - k3_s; \
	b = k1_s + k2_s; \

/* void calc_inv_complex(wk *a, wk *b){ */
/* 	wk c, d, aa, bb; */
/* 	aa = a[0]; bb = b[0]; */

/* 	c = aa; */
/* 	d = -bb; */

/* 	aa = (aa*aa + bb*bb); */

/* 	c /= aa; */
/* 	d /= aa; */
	
/* 	a[0] = c; */
/* 	b[0] = d; */
/* } */

/* void calc_inv_complex_scalar(wks *a, wks *b){ */
/* 	wks c, d, aa, bb; */
/* 	aa = a[0]; bb = b[0]; */

/* 	c = aa; */
/* 	d = -bb; */

/* 	aa = (aa*aa + bb*bb); */

/* 	c /= aa; */
/* 	d /= aa; */
	
/* 	a[0] = c; */
/* 	b[0] = d; */
/* } */


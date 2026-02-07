/**
 * =============================================================================
 * WEKUA OpenCL Type System Library
 * =============================================================================
 *
 * This library provides a unified type system for OpenCL kernels that supports
 * scalar, vector, and complex number operations across all numeric types.
 *
 * COMPILE-TIME PARAMETERS
 * -----------------------
 * WK_DTYPE        - Data type identifier (0-19), set during kernel compilation
 * WK_VECTOR_WIDTH - Vector width (1, 2, 4, 8, 16), determines SIMD operations
 * WK_COMPLEX      - Complex number flag (0 or 1), enables complex arithmetic
 * WK_DTYPE_ID     - Type family identifier (0-9), groups scalar/complex pairs
 *
 * TYPE DEFINITIONS
 * ----------------
 * T               - Base scalar type (char, short, int, long, float, double)
 * UT              - Unsigned variant of T (only for integer types)
 * wks             - Work scalar type (scalar or complex struct)
 * uwks            - Unsigned work scalar (for integer types)
 * wk              - Work vector type (matches WK_VECTOR_WIDTH)
 * wk2, wk4, wk8   - Fixed-width vector types (when supported by WK_VECTOR_WIDTH)
 *
 * MACROS
 * ------
 * WK_INT_MAX      - Maximum value for signed integer types
 * WK_UINT_MAX     - Maximum value for unsigned integer types
 * WKS_IS_UNSIGNED - Flag (0 or 1) indicating if wks is unsigned
 * convert_T       - Type conversion macro with saturation and rounding
 * COMPLEX_MUL_K   - Declares temporaries for complex multiplication
 * COMPLEX_MUL     - Performs complex multiplication using Karatsuba algorithm
 *
 * WK_DTYPE MAPPING
 * ----------------
 *  0: i8          10: Complex<i8>
 *  1: u8          11: Complex<u8>
 *  2: i16         12: Complex<i16>
 *  3: u16         13: Complex<u16>
 *  4: i32         14: Complex<i32>
 *  5: u32         15: Complex<u32>
 *  6: i64         16: Complex<i64>
 *  7: u64         17: Complex<u64>
 *  8: f32         18: Complex<f32>
 *  9: f64         19: Complex<f64>
 *
 * USAGE EXAMPLES
 * --------------
 * Scalar operation:     wks value = input[i];
 * Vector operation:     wk vec_data = vload(0, input);
 * Complex arithmetic:   COMPLEX_MUL(a, b, result);
 * Type conversion:      wks result = convert_T(input_value);
 * Range checking:       if (value > WK_INT_MAX) { ... }
 *
 * =============================================================================
 */

#if WK_DTYPE == 0

#define WK_INT_MAX CHAR_MAX
#define WK_UINT_MAX UCHAR_MAX

typedef char T;
typedef uchar uwks;
typedef char wks;

#define WKS_IS_UNSIGNED 0

#if WK_VECTOR_WIDTH >= 2
typedef char2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef char4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef char8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef char wk;
#elif WK_VECTOR_WIDTH == 2
typedef char2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef char4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef char8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef char16 wk;
#endif

#define convert_T convert_char_sat_rte

#elif WK_DTYPE == 1

#define WK_INT_MAX UCHAR_MAX
#define WK_UINT_MAX UCHAR_MAX

typedef uchar T;
typedef uchar uwks;
typedef uchar wks;

#define WKS_IS_UNSIGNED 1

#if WK_VECTOR_WIDTH >= 2
typedef uchar2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef uchar4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef uchar8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef uchar wk;
#elif WK_VECTOR_WIDTH == 2
typedef uchar2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef uchar4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef uchar8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef uchar16 wk;
#endif

#define convert_T convert_uchar_sat_rte

#elif WK_DTYPE == 2

#define WK_INT_MAX SHRT_MAX
#define WK_UINT_MAX USHRT_MAX

typedef short T;
typedef ushort uwks;
typedef short wks;

#define WKS_IS_UNSIGNED 0

#if WK_VECTOR_WIDTH >= 2
typedef short2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef short4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef short8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef short wk;
#elif WK_VECTOR_WIDTH == 2
typedef short2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef short4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef short8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef short16 wk;
#endif

#define convert_T convert_short_sat_rte

#elif WK_DTYPE == 3

#define WK_INT_MAX USHRT_MAX
#define WK_UINT_MAX USHRT_MAX

typedef ushort T;
typedef ushort uwks;
typedef ushort wks;

#define WKS_IS_UNSIGNED 1

#if WK_VECTOR_WIDTH >= 2
typedef ushort2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef ushort4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef ushort8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef ushort wk;
#elif WK_VECTOR_WIDTH == 2
typedef ushort2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef ushort4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef ushort8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef ushort16 wk;
#endif

#define convert_T convert_ushort_sat_rte

#elif WK_DTYPE == 4

#define WK_INT_MAX INT_MAX
#define WK_UINT_MAX UINT_MAX

typedef int T;
typedef uint uwks;
typedef int wks;

#define WKS_IS_UNSIGNED 0

#if WK_VECTOR_WIDTH >= 2
typedef int2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef int4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef int8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef int wk;
#elif WK_VECTOR_WIDTH == 2
typedef int2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef int4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef int8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef int16 wk;
#endif

#define convert_T convert_int_sat_rte

#elif WK_DTYPE == 5

#define WK_INT_MAX UINT_MAX
#define WK_UINT_MAX UINT_MAX

typedef uint T;
typedef uint uwks;
typedef uint wks;

#define WKS_IS_UNSIGNED 1

#if WK_VECTOR_WIDTH >= 2
typedef uint2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef uint4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef uint8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef uint wk;
#elif WK_VECTOR_WIDTH == 2
typedef uint2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef uint4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef uint8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef uint16 wk;
#endif

#define convert_T convert_uint_sat_rte

#elif WK_DTYPE == 6

#define WK_INT_MAX LONG_MAX
#define WK_UINT_MAX ULONG_MAX

typedef long T;
typedef ulong uwks;
typedef long wks;

#define WKS_IS_UNSIGNED 0

#if WK_VECTOR_WIDTH >= 2
typedef long2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef long4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef long8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef long wk;
#elif WK_VECTOR_WIDTH == 2
typedef long2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef long4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef long8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef long16 wk;
#endif

#define convert_T convert_long_sat_rte

#elif WK_DTYPE == 7

#define WK_INT_MAX ULONG_MAX
#define WK_UINT_MAX ULONG_MAX

typedef ulong T;
typedef ulong uwks;
typedef ulong wks;

#define WKS_IS_UNSIGNED 1

#if WK_VECTOR_WIDTH >= 2
typedef ulong2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef ulong4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef ulong8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef ulong wk;
#elif WK_VECTOR_WIDTH == 2
typedef ulong2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef ulong4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef ulong8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef ulong16 wk;
#endif

#define convert_T convert_ulong_sat_rte

#elif WK_DTYPE == 8

typedef float T;
typedef float wks;

#if WK_VECTOR_WIDTH >= 2
typedef float2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef float4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef float8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef float wk;
#elif WK_VECTOR_WIDTH == 2
typedef float2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef float4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef float8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef float16 wk;
#endif

#define convert_T convert_float

#elif WK_DTYPE == 9

typedef double T;
typedef double wks;

#if WK_VECTOR_WIDTH >= 2
typedef double2 wk2;
#endif

#if WK_VECTOR_WIDTH >= 4
typedef double4 wk4;
#endif

#if WK_VECTOR_WIDTH >= 8
typedef double8 wk8;
#endif

#if WK_VECTOR_WIDTH == 1
typedef double wk;
#elif WK_VECTOR_WIDTH == 2
typedef double2 wk;
#elif WK_VECTOR_WIDTH == 4
typedef double4 wk;
#elif WK_VECTOR_WIDTH == 8
typedef double8 wk;
#elif WK_VECTOR_WIDTH == 16
typedef double16 wk;
#endif

#define convert_T convert_double

#elif WK_DTYPE == 10

#define WK_INT_MAX CHAR_MAX
#define WK_UINT_MAX UCHAR_MAX
#define WKS_IS_UNSIGNED 0
typedef char T;
typedef uchar UT;
typedef __attribute__((packed)) struct {
    uchar real;
    uchar imag;
} uwks;
typedef __attribute__((packed)) struct {
    char real;
    char imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 11

#define WK_INT_MAX UCHAR_MAX
#define WK_UINT_MAX UCHAR_MAX
#define WKS_IS_UNSIGNED 1
typedef uchar T;
typedef uchar UT;
typedef __attribute__((packed)) struct {
    uchar real;
    uchar imag;
} uwks;
typedef __attribute__((packed)) struct {
    uchar real;
    uchar imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 12

#define WK_INT_MAX SHRT_MAX
#define WK_UINT_MAX USHRT_MAX
#define WKS_IS_UNSIGNED 0
typedef short T;
typedef ushort UT;
typedef __attribute__((packed)) struct {
    ushort real;
    ushort imag;
} uwks;
typedef __attribute__((packed)) struct {
    short real;
    short imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 13

#define WK_INT_MAX USHRT_MAX
#define WK_UINT_MAX USHRT_MAX
#define WKS_IS_UNSIGNED 1
typedef ushort T;
typedef ushort UT;
typedef __attribute__((packed)) struct {
    ushort real;
    ushort imag;
} uwks;
typedef __attribute__((packed)) struct {
    ushort real;
    ushort imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 14

#define WK_INT_MAX INT_MAX
#define WK_UINT_MAX UINT_MAX
#define WKS_IS_UNSIGNED 0
typedef int T;
typedef uint UT;
typedef __attribute__((packed)) struct {
    uint real;
    uint imag;
} uwks;
typedef __attribute__((packed)) struct {
    int real;
    int imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 15

#define WK_INT_MAX UINT_MAX
#define WK_UINT_MAX UINT_MAX
#define WKS_IS_UNSIGNED 1
typedef uint T;
typedef uint UT;
typedef __attribute__((packed)) struct {
    uint real;
    uint imag;
} uwks;
typedef __attribute__((packed)) struct {
    uint real;
    uint imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 16

#define WK_INT_MAX LONG_MAX
#define WK_UINT_MAX ULONG_MAX
#define WKS_IS_UNSIGNED 0
typedef long T;
typedef ulong UT;
typedef __attribute__((packed)) struct {
    ulong real;
    ulong imag;
} uwks;
typedef __attribute__((packed)) struct {
    long real;
    long imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 17

#define WK_INT_MAX ULONG_MAX
#define WK_UINT_MAX ULONG_MAX
#define WKS_IS_UNSIGNED 1
typedef ulong T;
typedef ulong UT;
typedef __attribute__((packed)) struct {
    ulong real;
    ulong imag;
} uwks;
typedef __attribute__((packed)) struct {
    ulong real;
    ulong imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 18

typedef float T;
typedef __attribute__((packed)) struct {
    float real;
    float imag;
} uwks;
typedef __attribute__((packed)) struct {
    float real;
    float imag;
} wks;
typedef wks wk;

#elif WK_DTYPE == 19

typedef double T;
typedef __attribute__((packed)) struct {
    double real;
    double imag;
} uwks;
typedef __attribute__((packed)) struct {
    double real;
    double imag;
} wks;
typedef wks wk;

#endif

#if WK_VECTOR_WIDTH > 1

/**
 * sum - Reduces a vector to a scalar by summing all elements
 * @a: Input vector of type wk
 *
 * Performs a tree-based reduction to sum all vector elements efficiently.
 * The reduction depth depends on WK_VECTOR_WIDTH (2->1 level, 4->2 levels, etc.)
 *
 * Returns: Scalar sum of all vector elements (type wks)
 */
inline wks sum(wk a) {
#if WK_VECTOR_WIDTH == 1
	return a;
#elif WK_VECTOR_WIDTH == 2
	return a.lo + a.hi;
#elif WK_VECTOR_WIDTH == 4
    const wk2 temp = a.lo + a.hi;
    return temp.lo + temp.hi;
#elif WK_VECTOR_WIDTH == 8
	const wk4 temp = a.lo + a.hi;
    const wk2 temp2 = temp.lo + temp.hi;
    return temp2.lo + temp2.hi;
#elif WK_VECTOR_WIDTH == 16
	const wk8 temp = a.lo + a.hi;
	const wk4 temp2 = temp.lo + temp.hi;
    const wk2 temp3 = temp2.lo + temp2.hi;
    return temp3.lo + temp3.hi;
#endif
}

#endif


#if WK_COMPLEX == 1

/**
 * COMPLEX_MUL_K - Declares temporary variables for complex multiplication
 * @T: Base scalar type for temporary variables
 *
 * Use this macro at the beginning of a function that performs complex multiplication
 * to declare the three temporary variables (k1, k2, k3) needed by COMPLEX_MUL.
 *
 * Example:
 *   COMPLEX_MUL_K(float)
 *   wks a, b, result;
 *   COMPLEX_MUL(a, b, result);
 */
#define COMPLEX_MUL_K(T) \
	T k1, k2, k3;

/**
 * COMPLEX_MUL - Multiplies two complex numbers using Karatsuba algorithm
 * @a: First complex number (has .real and .imag fields)
 * @b: Second complex number (has .real and .imag fields)
 * @res: Output complex number where result is stored
 *
 * Performs complex multiplication (a * b) using only 3 multiplications instead of 4
 * via the Karatsuba algorithm. Requires COMPLEX_MUL_K(T) to be called first to
 * declare temporary variables k1, k2, k3.
 *
 * Formula: (a.real + i*a.imag) * (b.real + i*b.imag)
 *   k1 = b.real * (a.real + a.imag)
 *   k2 = a.real * (b.imag - b.real)
 *   k3 = a.imag * (b.real + b.imag)
 *   res.real = k1 - k3
 *   res.imag = k1 + k2
 */
#define COMPLEX_MUL(a, b, res) \
	k1 = b.real*(a.real + a.imag); \
	k2 = a.real*(b.imag - b.real); \
	k3 = a.imag*(b.real + b.imag); \
	res.real = k1 - k3; \
	res.imag = k1 + k2;

#endif

/* #define COMPLEX_S_MUL_K(T) \ */
/* 	T k1_s, k2_s, k3_s; */

/* #define COMPLEX_S_MUL(a, b, c, d) \ */
/* 	k1_s = c*(a + b); \ */
/* 	k2_s = a*(d - c); \ */
/* 	k3_s = b*(c + d); \ */
/* 	a = k1_s - k3_s; \ */
/* 	b = k1_s + k2_s; \ */

#if dtype == 0

typedef char wks;

#if width == 1
typedef char wk;
#elif width == 2
typedef char2 wk;
#elif width == 4
typedef char4 wk;
#elif width == 8
typedef char8 wk;
#elif width == 16
typedef char16 wk;
#endif

#define convert_T convert_char_sat_rte

#elif dtype == 1

typedef unsigned char wks;

#if width == 1
typedef unsigned char wk;
#elif width == 2
typedef unsigned char2 wk;
#elif width == 4
typedef unsigned char4 wk;
#elif width == 8
typedef unsigned char8 wk;
#elif width == 16
typedef unsigned char16 wk;
#endif

#define convert_T convert_uchar_sat_rte

#elif dtype == 2

typedef short wks;

#if width == 1
typedef short wk;
#elif width == 2
typedef short2 wk;
#elif width == 4
typedef short4 wk;
#elif width == 8
typedef short8 wk;
#elif width == 16
typedef short16 wk;
#endif

#define convert_T convert_short_sat_rte

#elif dtype == 3

typedef unsigned short wks;

#if width == 1
typedef unsigned short wk;
#elif width == 2
typedef unsigned short2 wk;
#elif width == 4
typedef unsigned short4 wk;
#elif width == 8
typedef unsigned short8 wk;
#elif width == 16
typedef unsigned short16 wk;
#endif

#define convert_T convert_ushort_sat_rte

#elif dtype == 4

typedef int wks;

#if width == 1
typedef int wk;
#elif width == 2
typedef int2 wk;
#elif width == 4
typedef int4 wk;
#elif width == 8
typedef int8 wk;
#elif width == 16
typedef int16 wk;
#endif

#define convert_T convert_int_sat_rte

#elif dtype == 5

typedef unsigned int wks;

#if width == 1
typedef unsigned int wk;
#elif width == 2
typedef unsigned int2 wk;
#elif width == 4
typedef unsigned int4 wk;
#elif width == 8
typedef unsigned int8 wk;
#elif width == 16
typedef unsigned int16 wk;
#endif

#define convert_T convert_uint_sat_rte

#elif dtype == 6

typedef long wks;

#if width == 1
typedef long wk;
#elif width == 2
typedef long2 wk;
#elif width == 4
typedef long4 wk;
#elif width == 8
typedef long8 wk;
#elif width == 16
typedef long16 wk;
#endif

#define convert_T convert_long_sat_rte

#elif dtype == 7

typedef unsigned long wks;

#if width == 1
typedef unsigned long wk;
#elif width == 2
typedef unsigned long2 wk;
#elif width == 4
typedef unsigned long4 wk;
#elif width == 8
typedef unsigned long8 wk;
#elif width == 16
typedef unsigned long16 wk;
#endif

#define convert_T convert_ulong_sat_rte

#elif dtype == 8

typedef float wks;

#if width == 1
typedef float wk;
#elif width == 2
typedef float2 wk;
#elif width == 4
typedef float4 wk;
#elif width == 8
typedef float8 wk;
#elif width == 16
typedef float16 wk;
#endif

#define convert_T convert_float

#elif dtype == 9

typedef double wks;

#if width == 1
typedef double wk;
#elif width == 2
typedef double2 wk;
#elif width == 4
typedef double4 wk;
#elif width == 8
typedef double8 wk;
#elif width == 16
typedef double16 wk;
#endif

#define convert_T convert_double

#endif
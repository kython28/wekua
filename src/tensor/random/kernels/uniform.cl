#include "wekua.h"

#define bswap16(x) (x >> 8) | (x << 8)
#define bswap32(x) \
    ((x & 0x000000FF) << 24) | \
    ((x & 0x0000FF00) << 8) | \
    ((x & 0x00FF0000) >> 8) | \
    ((x & 0xFF000000) >> 24)

#define bswap64(x) \
    ((x & 0x00000000000000FFUL) << 56) | \
    ((x & 0x000000000000FF00UL) << 40) | \
    ((x & 0x0000000000FF0000UL) << 24) | \
    ((x & 0x00000000FF000000UL) << 8) | \
    ((x & 0x000000FF00000000UL) >> 8) | \
    ((x & 0x0000FF0000000000UL) >> 24) | \
    ((x & 0x00FF000000000000UL) >> 40) | \
    ((x & 0xFF00000000000000UL) >> 56)

#define rotl64(x, k) ( ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFFUL )

#if __ENDIAN_LITTLE__
#define FLIP_CONSTANT1 0x7C01812CF721AD1CUL
#define FLIP_CONSTANT2 0xDED46DE9839097DBUL
#else
#define FLIP_CONSTANT1 0x1CAD21CF1218017CUL
#define FLIP_CONSTANT2 0xDB979083E96DD4DEUL
#endif

#define PRIME_MULTIPLIER 0x9FB21C651E98DF25UL

ulong xxhash64(ulong index, ulong global_seed) {
#if __ENDIAN_LITTLE__
    const ulong seed2 = (global_seed & 0xFFFFFFFF00000000UL);
#else
    const ulong seed2 = (global_seed & 0x00000000FFFFFFFFUL);
#endif
    const ulong mixed = global_seed ^ (seed2 << 32);
    const ulong key = (FLIP_CONSTANT1 ^ FLIP_CONSTANT2) - mixed;

    const uint *blk = (uint *)&index;
#if __ENDIAN_LITTLE__
    const ulong WK_COMPLEXbined = (((ulong)blk[0]) << 32) + blk[1];
#else
    const ulong blk0 = bswap32(blk[0]);
    const ulong blk1 = bswap32(blk[1]);
    const ulong WK_COMPLEXbined = (((ulong)blk0) << 32) + blk1;
#endif

    const ulong x0 = WK_COMPLEXbined ^ key;
    const ulong x1 = x0 ^ rotl64(x0, 49) ^ rotl64(x0, 24) * PRIME_MULTIPLIER;
    const ulong x2 = x1 ^ ((x1 >> 35) + 8) * PRIME_MULTIPLIER;
    return x2 ^ (x2 >> 28);
}

__kernel void uniform(
    __global wks *restrict numbers,

	const ulong row_pitch,
    const ulong slice_pitch,

    const ulong global_seed

#if RANGE_DEFINED
    , const wks min_value,
    const wks max_value
#endif
) {
	const ulong i = get_global_id(0);
	const ulong j = get_global_id(1);
    const ulong k = get_global_id(2);

#if WK_COMPLEX
    const ulong index = i*slice_pitch + j*row_pitch + (k << 1);
#else
    const ulong index = i*slice_pitch + j*row_pitch + k;
#endif

#if RANGE_DEFINED

    // TODO: Find a way without using floating point
#ifdef cl_khr_fp64
    const double normalized = (double) xxhash64(index, global_seed) / ULONG_MAX;

#if WK_COMPLEX
    const double inormalized = (double) xxhash64(index + 1, global_seed) / ULONG_MAX;
#endif

#else
    const float normalized = (float) xxhash64(index, global_seed) / ULONG_MAX;

#if WK_COMPLEX
    const float inormalized = (float) xxhash64(index + 1, global_seed) / ULONG_MAX;
#endif

#endif

    numbers[index] = min_value + (wks)(normalized * (max_value - min_value + 1));
#if WK_COMPLEX
    numbers[index + 1] = min_value + (wks)(inormalized * (max_value - min_value + 1));
#endif

#else

#if WK_DTYPE >= 8

#if WK_COMPLEX
    numbers[index] = xxhash64(index, global_seed)/((wks)ULONG_MAX);
    numbers[index + 1] = xxhash64(index + 1, global_seed)/((wks)ULONG_MAX);
#else
    numbers[index] = xxhash64(index, global_seed)/((wks)ULONG_MAX);
#endif

#elif WK_DTYPE >= 6

#if WK_COMPLEX
    const uwks real_value = (uwks) xxhash64(index, global_seed);
    const uwks imag_value = (uwks) xxhash64(index + 1, global_seed);

#if WKS_IS_UNSIGNED
    numbers[index] = real_value;
    numbers[index + 1] = imag_value;
#else
    numbers[index] =  *((wks*)&real_value);
    numbers[index + 1] =  *((wks*)&imag_value);
#endif

#else
    const uwks real_value = (uwks) xxhash64(index, global_seed);

#if WKS_IS_UNSIGNED
    numbers[index] = real_value;
#else
    numbers[index] =  *((wks*)&real_value);
#endif

#endif

#else

#if WK_COMPLEX == 1
    const uwks real_value = (uwks) (xxhash64(index, global_seed) & WK_UINT_MAX);
    const uwks imag_value = (uwks) (xxhash64(index + 1, global_seed) & WK_UINT_MAX);

#if WKS_IS_UNSIGNED
    numbers[index] = real_value;
    numbers[index + 1] = imag_value;
#else
    numbers[index] =  *((wks*)&real_value);
    numbers[index + 1] =  *((wks*)&imag_value);
#endif

#else
    const uwks real_value = (uwks) (xxhash64(index, global_seed) & WK_UINT_MAX);

#if WKS_IS_UNSIGNED
    numbers[index] = real_value;
#else
    numbers[index] =  *((wks*)&real_value);
#endif

#endif

#endif

#endif
}

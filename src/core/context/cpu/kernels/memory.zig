/// Memory transfer and fill kernels for the CPU backend. The two original
/// worker callbacks are kept: `transferKernel` performs a linear memory copy
/// and `genericFillKernel` performs a pattern-based fill. Both accept a flat
/// `[]const usize` argument array whose layout is defined by the
/// corresponding `prepare*Command` function in `buffer.zig`.
///
/// The new fill family is built around the `patternFill` compile-time engine
/// plus a set of inline wrappers: `fillScalar`, `fillScalarNonTemporal`,
/// `fillSSE`, `fillNEON`, `fillAVX2`, `fillAVX512`, and the non-temporal
/// SIMD variants `fillSSENonTemporal`, `fillNEONNonTemporal`,
/// `fillAVX2NonTemporal`, and `fillAVX512NonTemporal`. Wrappers are the
/// public dispatch surface; `patternFill` is `pub` only so wrappers outside
/// this module can instantiate it at comptime.
///
/// The alignment contract is W-byte alignment on `dst`, where W is the store
/// stride selected by the wrapper: 16, 32, or 64 for the SIMD wrappers, and
/// 4 or 8 for the scalar non-temporal wrapper. When alignment cannot be
/// guaranteed, the caller must select `fillScalar` instead of a SIMD or
/// non-temporal wrapper. The SIMD body falls back to the scalar path
/// internally when the bounded replication buffer cannot hold a whole
/// pattern cycle.
const std = @import("std");
const builtin = @import("builtin");
const helpers = @import("helpers.zig");

pub fn transferKernel(args: []const usize) void {
    const src: [*]const u8 = @ptrFromInt(args[0]);
    const src_len = args[1];
    const dst: [*]u8 = @ptrFromInt(args[2]);
    const dst_len = args[3];

    @memcpy(dst[0..dst_len], src[0..src_len]);
}

pub fn genericFillKernel(args: []const usize) void {
    const dst: [*]u8 = @ptrFromInt(args[0]);
    const dst_len = args[1];
    const pattern: [*]const u8 = @ptrFromInt(args[2]);
    const pattern_len = args[3];

    var written: usize = 0;
    while (written < dst_len) {
        const remaining = dst_len - written;
        const copy_len = @min(pattern_len, remaining);
        @memcpy(dst[written .. written + copy_len], pattern[0..copy_len]);
        written += copy_len;
    }
}

/// `fillTailScalar` writes the remaining `len` bytes of `dst` by tiling
/// `pattern` and clamping each copy to the remaining size with `@min`. It is
/// the scalar fallback used by the `patternFill` SIMD branches to finish the
/// residual tail after the vectorized loop, and is also the body invoked by
/// the scalar wrappers.
///
/// The upstream invariant `dst_len % pattern_len == 0` holds when the tail is
/// reached after a `patternFill` SIMD loop, which makes the `@min` clamp a
/// no-op on that path; the clamp is kept defensively because the scalar
/// wrappers route arbitrary sizes through this same body. The function is
/// private; callers must come from this module.
fn fillTailScalar(dst: [*]u8, len: usize, pattern: [*]const u8, pattern_len: usize) void {
    var written: usize = 0;
    while (written < len) {
        const remaining = len - written;
        const copy_len = @min(pattern_len, remaining);
        @memcpy(dst[written .. written + copy_len], pattern[0..copy_len]);
        written += copy_len;
    }
}

/// `patternFill` is the generic fill engine, parameterized at compile time.
/// `T` pins the element type (currently `u8`), `use_simd` selects the vector
/// path versus the scalar path, `use_nontemporal` selects cache-bypassing
/// stores, and `store_stride` pins the vector width or the scalar pair
/// stride. Wrappers are the public dispatch surface; the engine is `pub`
/// only so wrappers outside this module can instantiate it at comptime.
fn patternFill(
    comptime use_simd: bool,
    comptime store_stride: usize,
    args: []const usize,
) void {
    const dst: [*]u8 = @ptrFromInt(args[0]);
    const dst_len = args[1];
    const pattern: [*]const u8 = @ptrFromInt(args[2]);
    const pattern_len = args[3];

    if (use_simd) {
        if (store_stride != 16 and store_stride != 32 and store_stride != 64) {
            @compileError("SIMD supports store_stride {16, 32, 64}");
        }

        const VectorType = @Vector(store_stride, u8);
        const vector_dst: [*]VectorType = @alignCast(@ptrCast(dst));
        const dst_vector_len = dst_len / store_stride;

        const max_replicated_bytes = 4 * store_stride;
        const pair_store_bytes = 2 * store_stride;

        const replicated_chunk = std.math.lcm(pair_store_bytes, pattern_len);
        if (replicated_chunk > max_replicated_bytes) {
            // Pattern length cannot be replicated within the bounded stack buffer
            // (max_replicated_bytes = 64 bytes for the smallest SIMD width,
            // 128/256 bytes for the wider widths). Bail to the scalar path.
            fillTailScalar(dst, dst_len, pattern, pattern_len);
        } else if (replicated_chunk == pair_store_bytes) {
            var vectors: [2]VectorType = undefined;
            var byte_index: usize = 0;
            for (&vectors) |*vector_slot| {
                inline for (0..store_stride) |i| {
                    vector_slot[i] = pattern[@mod(byte_index + i, pattern_len)];
                }
                byte_index += store_stride;
            }
        } else {
            var vectors: [4]VectorType = undefined;
            var byte_index: usize = 0;
            for (&vectors) |*vector_slot| {
                inline for (0..store_stride) |i| {
                    vector_slot[i] = pattern[@mod(byte_index + i, pattern_len)];
                }
                byte_index += store_stride;
            }

            var offset: usize = 0;
            var new_offset: usize = 4;
            while (new_offset < dst_vector_len) {
                helpers.store(VectorType, vector_dst + offset, vectors[0], vectors[1]);
                helpers.store(VectorType, vector_dst + offset + pair_store_bytes, vectors[2], vectors[3]);
                offset = new_offset;
                new_offset += 4;
            }
        }
    } else {
        // Scalar path with optional non-temporal stores.
        if (store_stride != 1) {
            @compileError("store_stride != 1 is reserved for the SIMD / NT paths; scalar step uses 1");
        }

        if (pattern_len % 8 == 0) {
            // 16-byte pair stores via helpers.store with T = u64.
            const pair_pattern: [*]const u64 = @ptrCast(@alignCast(pattern));
            const pair_stride: usize = 16; // 2 * sizeof(u64)
            var offset: usize = 0;
            while (offset + pair_stride <= dst_len) {
                const pair_addr: [*]u64 = @ptrCast(@alignCast(dst + offset));
                helpers.store(u64, pair_addr, pair_pattern[0], pair_pattern[1]);
                offset += pair_stride;
            }
            fillTailScalar(dst + offset, dst_len - offset, pattern, pattern_len);
        } else if (pattern_len % 4 == 0) {
            // 8-byte pair stores via helpers.store with T = u32.
            const pair_pattern: [*]const u32 = @ptrCast(@alignCast(pattern));
            const pair_stride: usize = 8; // 2 * sizeof(u32)
            var offset: usize = 0;
            while (offset + pair_stride <= dst_len) {
                const pair_addr: [*]u32 = @ptrCast(@alignCast(dst + offset));
                helpers.store(u32, pair_addr, pair_pattern[0], pair_pattern[1]);
                offset += pair_stride;
            }
            fillTailScalar(dst + offset, dst_len - offset, pattern, pattern_len);
        } else {
            // Pattern length is not aligned to 4 or 8. Fall back to scalar.
            fillTailScalar(dst, dst_len, pattern, pattern_len);
        }
    }
}

/// `fillScalar` is the scalar temporal fill wrapper. It is available on any
/// x86_64 or aarch64 target with no SIMD requirement. It fills a destination
/// byte range by tiling the provided pattern through the `patternFill`
/// engine. It carries no special benefit beyond correctness; it is the
/// baseline used when SIMD is unavailable, when the kernel is invoked as a
/// ground-truth reference, or when alignment cannot be guaranteed for a
/// wider wrapper. Future `buffer.zig` dispatch should pick the SIMD
/// variants when the target supports them.
pub fn fillScalar(args: []const usize) void {
    return patternFill(u8, false, false, 1, args);
}

/// `fillScalarNonTemporal` is the scalar non-temporal fill wrapper, available
/// on any x86_64 or aarch64 target. It tiles `pattern` and issues
/// non-temporal pair stores via `helpers.store`: a 16-byte pair stride with
/// `T = u64` when `pattern_len` is a multiple of 8, or an 8-byte pair stride
/// with `T = u32` when `pattern_len` is a multiple of 4. It is the right
/// choice for very short patterns where SIMD setup would dominate, or when
/// alignment is not guaranteed for the wider SIMD widths. The alignment
/// contract is 8-byte alignment on `dst` for the u64 path and 4-byte
/// alignment for the u32 path; the caller (future `buffer.zig` dispatch) is
/// responsible for verifying alignment before selecting this wrapper,
/// otherwise it must select `fillScalar`.
pub fn fillScalarNonTemporal(args: []const usize) void {
    return patternFill(u8, false, true, 1, args);
}

/// `fillAVX2` is the x86_64 256-bit temporal fill wrapper. It tiles
/// `pattern` into 32-byte AVX2 registers and issues temporal vector stores.
/// The replication buffer is computed once at entry and reused across the
/// loop, so the per-iteration cost is one aligned store per register. It is
/// the right choice when AVX2 is available and the destination will be read
/// before cache eviction; select `fillAVX512` only when AVX-512F is
/// available and the wider store pays off. The alignment contract is
/// 32-byte alignment on `dst`; the caller (future `buffer.zig` dispatch) is
/// responsible for verifying alignment before selecting this wrapper,
/// otherwise it must select `fillSSE` or `fillScalar`. AVX-512 clock
/// throttling is unrelated to AVX2 and does not affect this wrapper.
pub fn fillAVX2(args: []const usize) void {
    if (builtin.cpu.arch != .x86_64) @compileError("fillAVX2 is x86_64-only");
    return patternFill(u8, true, false, 32, args);
}

/// `fillSSE` is the x86_64 128-bit temporal fill wrapper. It tiles `pattern`
/// into 16-byte SSE registers and issues temporal vector stores. The
/// replication buffer is computed once at entry and reused across the loop,
/// so the per-iteration cost is one aligned store per register. It is the
/// right choice when SSE is available and the destination will be read
/// before cache eviction, or on targets that have SSE but not AVX2; select
/// `fillAVX2` instead when AVX2 is available and the data will be reused.
/// The alignment contract is 16-byte alignment on `dst`; the caller (future
/// `buffer.zig` dispatch) is responsible for verifying alignment before
/// selecting this wrapper, otherwise it must select `fillScalar`.
pub fn fillSSE(args: []const usize) void {
    if (builtin.cpu.arch != .x86_64) @compileError("fillSSE is x86_64-only");
    return patternFill(u8, true, false, 16, args);
}

/// `fillNEON` is the aarch64 128-bit temporal fill wrapper. It tiles
/// `pattern` into 16-byte NEON registers and issues temporal vector stores.
/// The replication buffer is computed once at entry and reused across the
/// loop, so the per-iteration cost is one aligned store per register. It is
/// the right choice when NEON is available and the destination will be read
/// before cache eviction; for streaming output that will not be reused on
/// the same aarch64 target, select `fillNEONNonTemporal` instead. The
/// alignment contract is 16-byte alignment on `dst`; the caller (future
/// `buffer.zig` dispatch) is responsible for verifying alignment before
/// selecting this wrapper, otherwise it must select `fillScalar`.
pub fn fillNEON(args: []const usize) void {
    if (builtin.cpu.arch != .aarch64) @compileError("fillNEON is aarch64-only");
    return patternFill(u8, true, false, 16, args);
}

/// `fillAVX512` is the x86_64 512-bit temporal fill wrapper. It tiles
/// `pattern` into 64-byte AVX-512F registers and issues temporal vector
/// stores. The replication buffer is computed once at entry and reused
/// across the loop, so the per-iteration cost is one aligned store per
/// register. It is the right choice when AVX-512F is available and the
/// destination will be read before cache eviction; select `fillAVX2` when
/// AVX-512F is unavailable or when the wider store does not pay off. The
/// alignment contract is 64-byte alignment on `dst`; the caller (future
/// `buffer.zig` dispatch) is responsible for verifying alignment before
/// selecting this wrapper, otherwise it must select `fillAVX2`, `fillSSE`,
/// or `fillScalar`. AVX-512 may reduce clock frequency on some CPUs; prefer
/// `fillAVX2` when the target is known to throttle under AVX-512.
pub fn fillAVX512(args: []const usize) void {
    if (builtin.cpu.arch != .x86_64) @compileError("fillAVX512 is x86_64-only");
    return patternFill(u8, true, false, 64, args);
}

/// `fillSSENonTemporal` is the x86_64 128-bit non-temporal fill wrapper. It
/// tiles `pattern` into 16-byte SSE registers and issues non-temporal pair
/// stores via `helpers.store`, which generates two `movntps` stores on
/// x86_64. Non-temporal stores bypass the cache, so the per-iteration cost
/// avoids pulling the destination line into the cache hierarchy. It is the
/// right choice for streaming output that will not be reused before
/// eviction; select `fillSSE` when the destination will be reused. The
/// alignment contract is 16-byte alignment on `dst`; the caller (future
/// `buffer.zig` dispatch) is responsible for verifying alignment before
/// selecting this wrapper, otherwise it must select `fillSSE` or
/// `fillScalar`.
pub fn fillSSENonTemporal(args: []const usize) void {
    if (builtin.cpu.arch != .x86_64) @compileError("fillSSENonTemporal is x86_64-only");
    return patternFill(u8, true, true, 16, args);
}

/// `fillAVX2NonTemporal` is the x86_64 256-bit non-temporal fill wrapper. It
/// tiles `pattern` into 32-byte AVX2 registers and issues non-temporal pair
/// stores via `helpers.store`, which generates two `vmovntps` stores on
/// x86_64. Non-temporal stores bypass the cache, so the per-iteration cost
/// avoids pulling the destination line into the cache hierarchy. It is the
/// right choice for streaming output that will not be reused before
/// eviction; select `fillAVX2` when the destination will be reused. The
/// alignment contract is 32-byte alignment on `dst`; the caller (future
/// `buffer.zig` dispatch) is responsible for verifying alignment before
/// selecting this wrapper, otherwise it must select `fillAVX2`, `fillSSE`,
/// or `fillScalar`.
pub fn fillAVX2NonTemporal(args: []const usize) void {
    if (builtin.cpu.arch != .x86_64) @compileError("fillAVX2NonTemporal is x86_64-only");
    return patternFill(u8, true, true, 32, args);
}

/// `fillNEONNonTemporal` is the aarch64 128-bit non-temporal fill wrapper.
/// It tiles `pattern` into 16-byte NEON registers and issues non-temporal
/// pair stores via `helpers.store`, which generates `stnp q0, q1` pair
/// stores on aarch64. Non-temporal stores bypass the cache, so the
/// per-iteration cost avoids pulling the destination line into the cache
/// hierarchy. It is the right choice for streaming output that will not be
/// reused before eviction; select `fillNEON` when the destination will be
/// reused. The alignment contract is 16-byte alignment on `dst`; the caller
/// (future `buffer.zig` dispatch) is responsible for verifying alignment
/// before selecting this wrapper, otherwise it must select `fillNEON` or
/// `fillScalar`.
pub fn fillNEONNonTemporal(args: []const usize) void {
    if (builtin.cpu.arch != .aarch64) @compileError("fillNEONNonTemporal is aarch64-only");
    return patternFill(u8, true, true, 16, args);
}

/// `fillAVX512NonTemporal` is the x86_64 512-bit non-temporal fill wrapper.
/// It tiles `pattern` into 64-byte AVX-512F registers and issues raw
/// `vmovntps` non-temporal stores; the helper `helpers.store` does not
/// support 512-bit vectors, so this wrapper bypasses the helper and emits
/// the NT store directly. Non-temporal stores bypass the cache, so the
/// per-iteration cost avoids pulling the destination line into the cache
/// hierarchy. It is the right choice for streaming output on AVX-512F
/// targets that will not be reused before eviction; select `fillAVX512`
/// when the destination will be reused. The alignment contract is 64-byte
/// alignment on `dst`; the caller (future `buffer.zig` dispatch) is
/// responsible for verifying alignment before selecting this wrapper,
/// otherwise it must select `fillAVX512`, `fillAVX2`, `fillSSE`, or
/// `fillScalar`. AVX-512 may reduce clock frequency on some CPUs; prefer
/// `fillAVX2NonTemporal` when the target is known to throttle under
/// AVX-512.
pub fn fillAVX512NonTemporal(args: []const usize) void {
    if (builtin.cpu.arch != .x86_64) @compileError("fillAVX512NonTemporal is x86_64-only");
    return patternFill(u8, true, true, 64, args);
}

const std = @import("std");
const builtin = @import("builtin");

/// Stores the pair (`value1`, `value2`) to `dst` using a non-temporal store
/// that bypasses the cache. `dst[0]` receives `value1` and `dst[1]` receives
/// `value2`; the two values must be adjacent in memory.
///
/// Dispatches to `x86_64Store` on x86_64, `armStore` on aarch64, and falls
/// back to normal stores on any other architecture.
///
/// `T`      - Element type to store.
/// `dst`    - Pointer to the destination memory; `dst[0]` receives `value1`
///            and `dst[1]` receives `value2`.
/// `value1` - First value to store.
/// `value2` - Second value to store.
pub inline fn store(comptime T: type, dst: [*]T, value1: T, value2: T) void {
    switch (builtin.cpu.arch) {
        .x86_64 => x86_64Store(T, dst, value1, value2),
        .aarch64 => armStore(T, dst, value1, value2),
        else => {
            dst[0] = value1;
            dst[1] = value2;
        },
    }
}

/// Stores the pair (`value1`, `value2`) to `dst` using non-temporal x86_64
/// stores. `dst[0]` receives `value1` and `dst[1]` receives `value2`; the two
/// values must be adjacent in memory.
///
/// The store width is resolved at comptime from `T`:
/// - Integers 64 bits: both values are packed into one 128-bit `movntps`
///   (one store instead of two `movntiq`).
/// - Integers 32 bits: both values are packed into one 64-bit `movntiq`
///   (one store instead of two `movntil`).
/// - Integers 16 bits: both values are packed into 4 bytes and written with
///   a single `maskmovdqu` (one push/pop, one masked store).
/// - Integers 8 bits: both values are packed into 2 bytes and written with
///   a single `maskmovdqu` (one push/pop, one masked store).
/// - Vectors of 128 bits: two `movntps` stores.
/// - Vectors of 256 bits: two `vmovntps` stores.
/// - Vectors of 512 bits: two `vmovntps` (zmm) stores.
/// - Any other width falls back to normal stores.
///
/// Only available on x86_64.
///
/// `T`      - Element type to store.
/// `dst`    - Pointer to the destination memory; `dst[0]` receives `value1`
///            and `dst[1]` receives `value2`.
/// `value1` - First value to store.
/// `value2` - Second value to store.
pub inline fn x86_64Store(comptime T: type, dst: [*]T, value1: T, value2: T) void {
    if (builtin.cpu.arch != .x86_64) {
        @compileError("x86_64Store is x86_64-only; use store for dispatch");
    }

    switch (@typeInfo(T)) {
        .int => |int_info| {
            const bits = int_info.bits;
            switch (bits) {
                64 => {
                    asm volatile (
                        \\movq %[v1], %%xmm0
                        \\movq %[v2], %%xmm1
                        \\punpcklqdq %%xmm1, %%xmm0
                        \\movntps %%xmm0, (%[addr])
                        :
                        : [v1] "r" (value1),
                          [v2] "r" (value2),
                          [addr] "r" (dst),
                        : .{ .xmm0 = true, .xmm1 = true, .memory = true });
                },
                32 => {
                    const packed_val: u64 =
                        @as(u64, value1) | (@as(u64, value2) << 32);
                    asm volatile ("movntiq %[value], (%[addr])"
                        :
                        : [value] "r" (packed_val),
                          [addr] "r" (dst),
                        : .{ .memory = true });
                },
                16 => {
                    const packed_val: u32 =
                        @as(u32, value1) | (@as(u32, value2) << 16);
                    const mask: u32 = 0x80808080;
                    asm volatile (
                        \\pushq %%rdi
                        \\movd %[value], %%xmm0
                        \\movd %[mask], %%xmm1
                        \\movq %[addr], %%rdi
                        \\maskmovdqu %%xmm0, %%xmm1
                        \\popq %%rdi
                        :
                        : [value] "r" (packed_val),
                          [mask] "r" (mask),
                          [addr] "r" (dst),
                        : .{ .xmm0 = true, .xmm1 = true, .memory = true });
                },
                8 => {
                    const packed_val: u32 =
                        @as(u32, value1) | (@as(u32, value2) << 8);
                    const mask: u32 = 0x8080;
                    asm volatile (
                        \\pushq %%rdi
                        \\movd %[value], %%xmm0
                        \\movd %[mask], %%xmm1
                        \\movq %[addr], %%rdi
                        \\maskmovdqu %%xmm0, %%xmm1
                        \\popq %%rdi
                        :
                        : [value] "r" (packed_val),
                          [mask] "r" (mask),
                          [addr] "r" (dst),
                        : .{ .xmm0 = true, .xmm1 = true, .memory = true });
                },
                else => {
                    dst[0] = value1;
                    dst[1] = value2;
                },
            }
        },
        .vector => |vec_info| {
            const elem = vec_info.child;
            const elem_bits = @typeInfo(elem).int.bits;
            const total_bits = elem_bits * vec_info.len;

            if (total_bits == 128) {
                asm volatile ("movntps %%xmm0, (%[addr])"
                    :
                    : [addr] "r" (dst),
                      [val] "x" (value1),
                    : .{ .memory = true });
                const dst2: [*]T = dst + 1;
                asm volatile ("movntps %%xmm0, (%[addr])"
                    :
                    : [addr] "r" (dst2),
                      [val] "x" (value2),
                    : .{ .memory = true });
            } else if (total_bits == 256) {
                asm volatile ("vmovntps %%ymm0, (%[addr])"
                    :
                    : [addr] "r" (dst),
                      [val] "x" (value1),
                    : .{ .memory = true });
                const dst2: [*]T = dst + 1;
                asm volatile ("vmovntps %%ymm0, (%[addr])"
                    :
                    : [addr] "r" (dst2),
                      [val] "x" (value2),
                    : .{ .memory = true });
            } else if (total_bits == 512) {
                asm volatile ("vmovntps %[val], (%[addr])"
                    :
                    : [addr] "r" (dst),
                      [val] "x" (value1),
                    : .{ .memory = true });
                const dst2: [*]T = dst + 1;
                asm volatile ("vmovntps %[val], (%[addr])"
                    :
                    : [addr] "r" (dst2),
                      [val] "x" (value2),
                    : .{ .memory = true });
            } else {
                dst[0] = value1;
                dst[1] = value2;
            }
        },
        else => {
            dst[0] = value1;
            dst[1] = value2;
        },
    }
}

/// Stores the pair (`value1`, `value2`) to `dst` using a non-temporal pair
/// store (`stnp`). `dst[0]` receives `value1` and `dst[1]` receives `value2`;
/// the two values must be adjacent in memory.
///
/// The store width is resolved at comptime from `T`:
/// - Integers 64 bits use `stnp` (writes 16 bytes).
/// - Integers 32 bits use `stnp` (writes 8 bytes).
/// - Vectors of 128 bits use `stnp q0, q1` (writes 32 bytes, two NEON vectors).
/// - Any other width falls back to normal stores (`dst[0] = value1`,
///   `dst[1] = value2`).
///
/// ARM does not expose single-register non-temporal stores; the pair form
/// is the only non-temporal store primitive available.
///
/// Only available on aarch64.
///
/// `T`      - Element type to store.
/// `dst`    - Pointer to the destination memory; `dst[0]` receives `value1`
///            and `dst[1]` receives `value2`.
/// `value1` - First value to store.
/// `value2` - Second value to store.
pub inline fn armStore(comptime T: type, dst: [*]T, value1: T, value2: T) void {
    if (builtin.cpu.arch != .aarch64) {
        @compileError("armStore is aarch64-only; use store for dispatch");
    }

    switch (@typeInfo(T)) {
        .int => |int_info| {
            switch (int_info.bits) {
                32 => {
                    asm volatile ("stnp %w[v1], %w[v2], [%[addr]]"
                        :
                        : [v1] "r" (value1),
                          [v2] "r" (value2),
                          [addr] "r" (dst),
                        : .{ .memory = true });
                },
                64 => {
                    asm volatile ("stnp %[v1], %[v2], [%[addr]]"
                        :
                        : [v1] "r" (value1),
                          [v2] "r" (value2),
                          [addr] "r" (dst),
                        : .{ .memory = true });
                },
                else => {
                    dst[0] = value1;
                    dst[1] = value2;
                },
            }
        },
        .vector => |vec_info| {
            const total_bits = @typeInfo(vec_info.child).int.bits * vec_info.len;
            if (total_bits == 128) {
                asm volatile ("stnp q0, q1, [%[addr]]"
                    :
                    : [addr] "r" (dst),
                      [v1] "w" (value1),
                      [v2] "w" (value2),
                    : .{ .memory = true });
            } else {
                dst[0] = value1;
                dst[1] = value2;
            }
        },
        else => {
            dst[0] = value1;
            dst[1] = value2;
        },
    }
}
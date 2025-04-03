const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

inline fn get_random_number(comptime T: type, randprg: std.Random) T {
    if (T == f32 or T == f64) {
        return -1 + randprg.float(T) * 2;
    } else {
        return randprg.int(T);
    }
}

inline fn complex_mul(comptime T: type, re1: T, im1: T, re2: T, im2: T) struct { T, T } {
    const k1: T = re2 * (re1 + im1);
    const k2: T = re1 * (im2 - re2);
    const k3: T = im1 * (re2 + im2);
    return .{ k1 - k3, k1 + k2 };
}

fn test_axpy(
    comptime T: type,
    allocator: std.mem.Allocator,
    ctx: *wekua.core.Context,
    randprg: std.Random,
    comptime is_complex: bool,
    comptime vectors_enabled: bool,
    comptime vectors_enabled2: bool,
) !void {
    const w_cmd = &ctx.command_queues[0];
    if (!w_cmd.typeIsSupported(T)) {
        return;
    }

    var shape: [4]u64 = .{
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
    };

    const tensor = try wekua.Tensor(T).alloc(ctx, &shape, .{
        .is_complex = is_complex,
        .vectors_enabled = vectors_enabled,
    });
    defer tensor.release();

    const tensor2 = try wekua.Tensor(T).alloc(ctx, &shape, .{
        .is_complex = is_complex,
        .vectors_enabled = vectors_enabled2,
    });
    defer tensor2.release();

    try wekua.tensor.random.uniform(T, w_cmd, tensor, 0, null, null);
    try wekua.tensor.random.uniform(T, w_cmd, tensor2, 0, null, null);

    const numbers2: []T = try allocator.alloc(T, tensor2.dimensions.number_of_elements_without_padding);
    defer allocator.free(numbers2);

    try wekua.tensor.memory.writeToBuffer(T, tensor2, w_cmd, numbers2);

    const alpha: T = get_random_number(T, randprg);
    const beta: ?T = if (is_complex) get_random_number(T, randprg) else null;

    try wekua.blas.axpy(T, w_cmd, tensor, alpha, beta, tensor2);

    const numbers1: []T = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers3: []T = try allocator.alloc(T, tensor2.dimensions.number_of_elements_without_padding);
    defer allocator.free(numbers3);

    try wekua.tensor.memory.writeToBuffer(T, tensor, w_cmd, numbers1);
    try wekua.tensor.memory.writeToBuffer(T, tensor2, w_cmd, numbers3);
    if (is_complex) {
        const eps: T = @floatCast(comptime std.math.floatEps(f32));
        for (0..(tensor.dimensions.number_of_elements_without_padding / 2)) |i| {
            const index = i * 2;
            var expected = complex_mul(T, numbers1[index], numbers1[index + 1], alpha, beta.?);
            expected[0] += numbers2[index];
            expected[1] += numbers2[index + 1];

            try std.testing.expectApproxEqAbs(expected[0], numbers3[index], eps);
            try std.testing.expectApproxEqAbs(expected[1], numbers3[index + 1], eps);
        }
    } else {
        for (numbers1, numbers2, numbers3) |n1, n2, n3| {
            switch (@typeInfo(T)) {
                .int => {
                    const expected: T = n1 *% alpha +% n2;
                    try std.testing.expectEqual(expected, n3);
                },
                .float => {
                    const eps = comptime std.math.floatEps(T);
                    const expected: T = n1 * alpha + n2;
                    try std.testing.expectApproxEqAbs(expected, n3, eps);
                },
                else => unreachable
            }
        }
    }
}

test {
    const allocator = std.testing.allocator;
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        try test_axpy(T, allocator, ctx, randprg, false, false, false);
        try test_axpy(T, allocator, ctx, randprg, false, false, true);
        try test_axpy(T, allocator, ctx, randprg, false, true, false);
        try test_axpy(T, allocator, ctx, randprg, false, true, true);

        if (T == f32 or T == f64) {
            // Hard to test with integers due overflow :-/
            try test_axpy(T, allocator, ctx, randprg, true, false, false);
            try test_axpy(T, allocator, ctx, randprg, true, false, true);
            try test_axpy(T, allocator, ctx, randprg, true, true, false);
            try test_axpy(T, allocator, ctx, randprg, true, true, true);
        }
    }
}

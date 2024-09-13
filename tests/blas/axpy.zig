const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

inline fn get_random_number(comptime T: type, randprg: std.Random) T {
    if (T == f32 or T == f64) {
        return randprg.float(T);
    }else{
        return randprg.int(T);
    }
}

inline fn complex_mul(comptime T: type, re1: T, im1: T, re2: T, im2: T) struct {T, T} {
    const k1: T = re2 * (re1 + im1);
    const k2: T = re1 * (im2 - re2);
    const k3: T = im1 * (re2 + im2);
    return .{k1 - k3, k1 + k2};
}

fn test_axpy(
    allocator: std.mem.Allocator, ctx: wekua.context.wContext, randprg: std.Random,
    comptime T: type, comptime is_complex: bool
) !void {
    var shape: [4]u64 = .{
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20)
    };

    const dtype = comptime wekua.tensor.get_wekua_dtype_from_zig_type(T);
    const tensor = try wekua.tensor.alloc(ctx, &shape, .{
        .dtype = dtype,
        .is_complex = is_complex
    });
    defer wekua.tensor.release(tensor);

    const tensor2 = try wekua.tensor.alloc(ctx, &shape, .{
        .dtype = dtype,
        .is_complex = is_complex
    });
    defer wekua.tensor.release(tensor2);

    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.extra.random.random(w_cmd, tensor);
    try wekua.tensor.extra.random.random(w_cmd, tensor2);

    const numbers2: []T = try allocator.alloc(T, tensor2.number_of_elements_without_padding);
    defer allocator.free(numbers2);

    try wekua.tensor.io.write_to_buffer(w_cmd, tensor2, numbers2);

    const alpha = get_random_number(T, randprg);
    const beta = get_random_number(T, randprg);

    const alpha_scalar: wekua.tensor.wScalar = wekua.tensor.create_scalar(alpha);
    var beta_scalar: ?wekua.tensor.wScalar = null;
    if (is_complex) {
        beta_scalar = wekua.tensor.create_scalar(beta);
    }

    try wekua.blas.axpy(w_cmd, tensor, alpha_scalar, beta_scalar, tensor2);

    const numbers1: []T = try allocator.alloc(T, tensor.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers3: []T = try allocator.alloc(T, tensor2.number_of_elements_without_padding);
    defer allocator.free(numbers3);

    try wekua.tensor.io.write_to_buffer(w_cmd, tensor, numbers1);
    try wekua.tensor.io.write_to_buffer(w_cmd, tensor2, numbers3);
    if (is_complex) {
        for (0..(tensor.number_of_elements_without_padding/2)) |i| {
            const index = i*2;
            var expected = complex_mul(T, numbers1[index], numbers1[index + 1], alpha, beta);
            expected[0] += numbers2[index];
            expected[1] += numbers2[index + 1];
            const eps = comptime std.math.floatEps(T);
            try std.testing.expectApproxEqAbs(expected[0], numbers3[index], eps);
            try std.testing.expectApproxEqAbs(expected[1], numbers3[index + 1], eps);
        }
    }else{
        for (numbers1, numbers2, numbers3) |n1, n2, n3| {
            if (T == f32 or T == f64) {
                const eps = comptime std.math.floatEps(T);
                const expected: T = n1 * alpha + n2;
                try std.testing.expectApproxEqAbs(expected, n3, eps);
            }else{
                const expected: T = n1 *% alpha +% n2;
                try std.testing.expectEqual(expected, n3);
            }
        }
    }
}

test {
    const allocator = std.testing.allocator;
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const types = .{u8, i8, u16, i16, u32, i32, u64, i64, f32, f64};
    const randprg = std.crypto.random;
    inline for (types) |T| {
        try test_axpy(allocator, ctx, randprg, T, false);
        if (T == f32 or T == f64) {
            // Hard to test with integers due overflow :-/
            try test_axpy(allocator, ctx, randprg, T, true);
        }
    }
}

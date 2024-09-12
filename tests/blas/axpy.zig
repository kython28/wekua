const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

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

    const alpha = blk: {
        if (T == f32 or T == f64) {
            break :blk randprg.float(T);
        }else{
            break :blk randprg.int(T);
        }
    };
    const alpha_scalar: wekua.tensor.wScalar = wekua.tensor.create_scalar(alpha);

    try wekua.blas.axpy(w_cmd, tensor, alpha_scalar, tensor2);

    const numbers1: []T = try allocator.alloc(T, tensor.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers3: []T = try allocator.alloc(T, tensor2.number_of_elements_without_padding);
    defer allocator.free(numbers3);

    try wekua.tensor.io.write_to_buffer(w_cmd, tensor, numbers1);
    try wekua.tensor.io.write_to_buffer(w_cmd, tensor2, numbers3);
    for (numbers1, numbers2, numbers3) |n1, n2, n3| {
        if (is_complex) {
            // TODO
        }else{
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
        // try test_axpy(allocator, ctx, randprg, T, true);
    }
}

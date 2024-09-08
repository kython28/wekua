
const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

const allocator = std.testing.allocator;

fn test_transpose(
    ctx: wekua.context.wContext, comptime is_complex: bool, randprg: std.Random,
    comptime T: type
) !void {
    var shape: [4]u64 = .{
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20)
    };

    const tensor = try wekua.tensor.alloc(ctx, &shape, .{
        .dtype = wekua.tensor.get_wekua_dtype_from_zig_type(T),
        .is_complex = is_complex
    });
    defer wekua.tensor.release(tensor);

    const w_cmd = ctx.command_queues[0];
    try wekua.tensor.extra.random.random(w_cmd, tensor);

    const dim0: u64 = randprg.intRangeAtMost(u64, 0, 2);
    const dim1: u64 = randprg.intRangeAtMost(u64, dim0+1, 3);

    const l_dim = shape[dim0];
    shape[dim0] = shape[dim1];
    shape[dim1] = l_dim;

    const tensor2 = try wekua.tensor.alloc(ctx, &shape, .{
        .dtype = wekua.tensor.get_wekua_dtype_from_zig_type(T),
        .is_complex = is_complex
    });
    defer wekua.tensor.release(tensor2);

    try wekua.tensor.extra.transpose(w_cmd, tensor2, tensor, dim0, dim1);

    const numbers1: []T = try allocator.alloc(T, tensor.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers2: []T = try allocator.alloc(T, tensor2.number_of_elements_without_padding);
    defer allocator.free(numbers2);

    try wekua.tensor.io.write_to_buffer(w_cmd, tensor, numbers1);
    try wekua.tensor.io.write_to_buffer(w_cmd, tensor2, numbers2);

    var multi_index: [4]u64 = undefined;
    const factor: u64 = (1 + @as(usize, @intFromBool(is_complex)));
    const number_of_elements: usize = tensor.number_of_elements_without_padding / factor;
    for (0..number_of_elements) |t1i| {
        wekua.utils.unravel_index(t1i * factor, tensor.shape, null, &multi_index, is_complex);

        const l_index = multi_index[dim0];
        multi_index[dim0] = multi_index[dim1];
        multi_index[dim1] = l_index;
        const t2i = wekua.utils.ravel_multi_index(&multi_index, tensor2.shape, null, is_complex);

        if (T == f32 or T == f64) {
            const eps = std.math.floatEps(T);
            try std.testing.expectApproxEqAbs(numbers1[t1i * factor], numbers2[t2i], eps);
            if (is_complex) {
                try std.testing.expectApproxEqAbs(numbers1[t1i * factor + 1], numbers2[t2i + 1], eps);
            }
        }else{
            try std.testing.expectEqual(numbers1[t1i * factor], numbers2[t2i]);
            if (is_complex) {
                try std.testing.expectEqual(numbers1[t1i * factor + 1], numbers2[t2i + 1]);
            }
        }
    }
}

test "Transpose and check" {
    const ctx = try wekua.context.create_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer wekua.context.release(ctx);

    const types = .{u8, i8, u16, i16, u32, i32, u64, i64, f32, f64};
    const randprg = std.crypto.random;
    inline for (types) |T| {
        try test_transpose(ctx, false, randprg, T);
        try test_transpose(ctx, true, randprg, T);
    }
}

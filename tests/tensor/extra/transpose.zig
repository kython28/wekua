const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

fn test_transpose(
    allocator: std.mem.Allocator,
    ctx: *const wekua.core.Context,
    comptime is_complex: bool,
    randprg: std.Random,
    comptime T: type,
) !void {
    var shape: [4]u64 = .{
        randprg.intRangeAtMost(u64, 1, 5),
        randprg.intRangeAtMost(u64, 1, 5),
        randprg.intRangeAtMost(u64, 1, 5),
        randprg.intRangeAtMost(u64, 1, 5),
    };

    std.log.warn("Shapers 1: {any}", .{shape});

    const tensor = try wekua.Tensor(T).alloc(ctx, &shape, .{ .is_complex = is_complex });
    defer tensor.release();

    const w_cmd = &ctx.command_queues[0];
    try wekua.tensor.random.fill(T, w_cmd, tensor, null);

    const dim0: u64 = randprg.intRangeAtMost(u64, 0, 2);
    const dim1: u64 = randprg.intRangeAtMost(u64, dim0 + 1, 3);

    const l_dim = shape[dim0];
    shape[dim0] = shape[dim1];
    shape[dim1] = l_dim;

    std.log.warn("Shapers 2: {any}", .{shape});

    const tensor2 = try wekua.Tensor(T).alloc(ctx, &shape, .{ .is_complex = is_complex });
    defer tensor2.release();

    try wekua.tensor.transpose(T, w_cmd, tensor2, tensor, dim0, dim1);

    const numbers1: []T = try allocator.alloc(T, tensor.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers2: []T = try allocator.alloc(T, tensor2.number_of_elements_without_padding);
    defer allocator.free(numbers2);

    try wekua.tensor.memory.writeToBuffer(T, tensor, w_cmd, numbers1);
    try wekua.tensor.memory.writeToBuffer(T, tensor2, w_cmd, numbers2);

    std.log.warn("{any}", .{numbers1});
    std.log.warn("{any}", .{numbers2});

    var multi_index: [4]u64 = undefined;
    const factor: u64 = (1 + @as(usize, @intFromBool(is_complex)));
    const number_of_elements: usize = tensor.number_of_elements_without_padding / factor;
    for (0..number_of_elements) |t1i| {
        wekua.utils.unravelIndex(t1i * factor, tensor.shape, null, &multi_index, is_complex);

        const l_index = multi_index[dim0];
        multi_index[dim0] = multi_index[dim1];
        multi_index[dim1] = l_index;
        const t2i = wekua.utils.ravelMultiIndex(
            &multi_index,
            tensor2.shape,
            null,
            is_complex,
        );

        switch (@typeInfo(T)) {
            .int => {
                try std.testing.expectEqual(numbers1[t1i * factor], numbers2[t2i]);
                if (is_complex) {
                    try std.testing.expectEqual(numbers1[t1i * factor + 1], numbers2[t2i + 1]);
                }
            },
            .float => {
                const eps = std.math.floatEps(T);
                try std.testing.expectApproxEqAbs(numbers1[t1i * factor], numbers2[t2i], eps);
                if (is_complex) {
                    try std.testing.expectApproxEqAbs(numbers1[t1i * factor + 1], numbers2[t2i + 1], eps);
                }
            },
            else => unreachable,
        }
    }
}

test "Transpose and check" {
    const allocator = std.testing.allocator;
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        try test_transpose(allocator, ctx, false, randprg, T);
        try test_transpose(allocator, ctx, true, randprg, T);
    }
}

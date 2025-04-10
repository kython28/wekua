const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

fn test_identity(
    comptime T: type,
    allocator: std.mem.Allocator,
    ctx: *const wekua.core.Context,
    randprg: std.Random,
    comptime is_complex: bool,
    comptime vectors_enabled: bool,
) !void {
    const w_cmd = &ctx.command_queues[0];
    if (!w_cmd.typeIsSupported(T)) {
        return;
    }

    const size = randprg.intRangeAtMost(u64, 10, 100);
    const shape: [3]u64 = .{
        size, size, size
    };

    const tensor = try wekua.Tensor(T).alloc(ctx, &shape, .{
        .is_complex = is_complex,
        .vectors_enabled = vectors_enabled
    });
    defer tensor.release();

    try wekua.tensor.identity(T, w_cmd, tensor);

    const numbers = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(numbers);

    try wekua.tensor.memory.writeToBuffer(T, tensor, w_cmd, numbers);

    var indices: [3]u64 = undefined;
    for (0..size) |i| {
        for (&indices) |*v| {
            v.* = i;
        }

        const index = wekua.utils.ravelMultiIndex(&indices, tensor.dimensions.shape, null, is_complex);
        switch (@typeInfo(T)) {
            .int => {
                try std.testing.expectEqual(numbers[index], 1);
                if (is_complex) {
                    try std.testing.expectEqual(numbers[index + 1], 0);
                }
            },
            .float => {
                const eps = comptime std.math.floatEps(T);
                try std.testing.expectApproxEqAbs(numbers[index], 1, eps);
                if (is_complex) {
                    try std.testing.expectApproxEqAbs(numbers[index + 1], 0, eps);
                }
            },
            else => unreachable
        }
    }
}

test "identity" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        try test_identity(T, allocator, ctx, randprg, false, false);
        try test_identity(T, allocator, ctx, randprg, false, true);

        try test_identity(T, allocator, ctx, randprg, true, false);
        try test_identity(T, allocator, ctx, randprg, true, true);
    }
}

const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

fn test_convertions_to_complex(
    comptime T: type,
    allocator: std.mem.Allocator,
    ctx: *const wekua.core.Context,
    randprg: std.Random,
    comptime dom: bool,
    comptime vectors_enabled: bool,
    comptime vectors_enabled2: bool,
) !void {
    var shape: [4]u64 = .{
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
    };

    const w_cmd = &ctx.command_queues[0];
    if (!w_cmd.typeIsSupported(T)) {
        return;
    }

    const tensor = try wekua.Tensor(T).alloc(ctx, &shape, .{
        .is_complex = false,
        .vectors_enabled = vectors_enabled,
    });
    defer tensor.release();

    try wekua.tensor.random.fill(T, w_cmd, tensor, 0);

    const tensor2 = try wekua.Tensor(T).alloc(ctx, &shape, .{
        .is_complex = true,
        .vectors_enabled = vectors_enabled2,
    });
    defer tensor2.release();

    try wekua.tensor.convertions.to_complex(T, w_cmd, tensor, tensor2, dom);

    const numbers1: []T = try allocator.alloc(T, tensor.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers2: []T = try allocator.alloc(T, tensor2.number_of_elements_without_padding);
    defer allocator.free(numbers2);

    try wekua.tensor.memory.writeToBuffer(T, tensor, w_cmd, numbers1);
    try wekua.tensor.memory.writeToBuffer(T, tensor2, w_cmd, numbers2);
    for (numbers1, 0..) |n1, index| {
        const n2 = numbers2[index * 2 + @intFromBool(dom)];
        const n3 = numbers2[index * 2 + (1 - @intFromBool(dom))];
        switch (@typeInfo(T)) {
            .int => {
                try std.testing.expectEqual(n1, n2);
                try std.testing.expectEqual(0, n3);
            },
            .float => {
                const eps = comptime std.math.floatEps(T);
                try std.testing.expectApproxEqAbs(n1, n2, eps);
                try std.testing.expectApproxEqAbs(0.0, n3, eps);
            },
            else => unreachable,
        }
    }
}

test "Convert real tensor to complex" {
    const allocator = std.testing.allocator;
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        try test_convertions_to_complex(T, allocator, ctx, randprg, false, false, false);
        try test_convertions_to_complex(T, allocator, ctx, randprg, false, true, false);
        try test_convertions_to_complex(T, allocator, ctx, randprg, false, false, true);
        try test_convertions_to_complex(T, allocator, ctx, randprg, false, true, true);

        try test_convertions_to_complex(T, allocator, ctx, randprg, true, false, false);
        try test_convertions_to_complex(T, allocator, ctx, randprg, true, true, false);
        try test_convertions_to_complex(T, allocator, ctx, randprg, true, false, true);
        try test_convertions_to_complex(T, allocator, ctx, randprg, true, true, true);
    }
}

fn test_convertions_to_real(
    comptime T: type,
    allocator: std.mem.Allocator,
    ctx: *const wekua.core.Context,
    randprg: std.Random,
    comptime dom: bool,
    comptime vectors_enabled: bool,
    comptime vectors_enabled2: bool,
) !void {
    var shape: [4]u64 = .{
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
        randprg.intRangeAtMost(u64, 1, 20),
    };

    const w_cmd = &ctx.command_queues[0];
    if (!w_cmd.typeIsSupported(T)) {
        return;
    }

    const tensor = try wekua.Tensor(T).alloc(ctx, &shape, .{
        .is_complex = true,
        .vectors_enabled = vectors_enabled,
    });
    defer tensor.release();

    try wekua.tensor.random.fill(T, w_cmd, tensor, 0);

    const tensor2 = try wekua.Tensor(T).alloc(ctx, &shape, .{
        .is_complex = false,
        .vectors_enabled = vectors_enabled2,
    });
    defer tensor2.release();

    try wekua.tensor.convertions.to_real(T, w_cmd, tensor, tensor2, dom);

    const numbers1: []T = try allocator.alloc(T, tensor.number_of_elements_without_padding);
    defer allocator.free(numbers1);

    const numbers2: []T = try allocator.alloc(T, tensor2.number_of_elements_without_padding);
    defer allocator.free(numbers2);

    try wekua.tensor.memory.writeToBuffer(T, tensor, w_cmd, numbers1);
    try wekua.tensor.memory.writeToBuffer(T, tensor2, w_cmd, numbers2);
    for (numbers2, 0..) |n1, index| {
        const n2 = numbers1[index * 2 + @intFromBool(dom)];
        switch (@typeInfo(T)) {
            .int => {
                try std.testing.expectEqual(n1, n2);
            },
            .float => {
                const eps = comptime std.math.floatEps(T);
                try std.testing.expectApproxEqAbs(n1, n2, eps);
            },
            else => unreachable,
        }
    }
}

test "Convert complex tensor to real" {
    const allocator = std.testing.allocator;
    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const randprg = std.crypto.random;
    inline for (wekua.core.SupportedTypes) |T| {
        try test_convertions_to_real(T, allocator, ctx, randprg, false, false, false);
        try test_convertions_to_real(T, allocator, ctx, randprg, false, true, false);
        try test_convertions_to_real(T, allocator, ctx, randprg, false, false, true);
        try test_convertions_to_real(T, allocator, ctx, randprg, false, true, true);

        try test_convertions_to_real(T, allocator, ctx, randprg, true, false, false);
        try test_convertions_to_real(T, allocator, ctx, randprg, true, true, false);
        try test_convertions_to_real(T, allocator, ctx, randprg, true, false, true);
        try test_convertions_to_real(T, allocator, ctx, randprg, true, true, true);
    }
}

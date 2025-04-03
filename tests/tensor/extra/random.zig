const wekua = @import("wekua");
const cl = @import("opencl");
const std = @import("std");

test "Create float tensor with random numbers" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const command_queue = &ctx.command_queues[0];

    inline for (wekua.core.SupportedTypes) |T| {
        const tensor = try wekua.Tensor(T).alloc(ctx, &.{ 100, 100, 100 }, .{});
        defer tensor.release();

        if (command_queue.typeIsSupported(T)) {
            try wekua.tensor.random.uniform(T, command_queue, tensor, null, null, null);

            const seed = std.crypto.random.int(u64);
            try wekua.tensor.random.uniform(T, command_queue, tensor, seed, null, null);
        }
    }
}

fn checkVariance(
    comptime T: type,
    allocator: std.mem.Allocator,
    command_queue: *const wekua.core.CommandQueue,
    tensor: *wekua.Tensor(T),
    min_value: ?T,
    max_value: ?T,
) !void {
    const buf = try allocator.alloc(T, tensor.dimensions.number_of_elements_without_padding);
    defer allocator.free(buf);

    try wekua.tensor.memory.writeToBuffer(T, tensor, command_queue, buf);

    var mean: f64 = 0;
    for (buf) |elem| {
        const val: f64 = switch (@typeInfo(T)) {
            .int => @floatFromInt(elem),
            .float => @floatCast(elem),
            else => unreachable,
        };

        mean += val;
    }

    const number_of_elements_float: f64 = @floatFromInt(tensor.dimensions.number_of_elements_without_padding);
    mean /= number_of_elements_float;

    var max_element: T = switch (@typeInfo(T)) {
        .int => std.math.minInt(T),
        .float => -std.math.floatMax(T),
        else => unreachable,
    };
    var min_element: T = switch (@typeInfo(T)) {
        .int => std.math.maxInt(T),
        .float => std.math.floatMax(T),
        else => unreachable,
    };

    var sq_diff: f64 = 0.0;
    for (buf) |elem| {
        const diff = switch (@typeInfo(T)) {
            .int => @as(f64, @floatFromInt(elem)) - mean,
            .float => @as(f64, @floatCast(elem)) - mean,
            else => unreachable,
        };
        sq_diff += diff * diff;

        max_element = @max(max_element, elem);
        min_element = @min(min_element, elem);
    }

    const variance: f64 = sq_diff / (number_of_elements_float - 1);

    const expected_variance = switch (@typeInfo(T)) {
        .int => blk: {
            const max_int: f64 = @floatFromInt(max_value orelse std.math.maxInt(T));
            const min_int: f64 = @floatFromInt(min_value orelse std.math.minInt(T));

            const diff = max_int - min_int;
            break :blk diff * diff / 12.0;
        },
        .float => blk: {
            const max_float: f64 = @floatCast(max_value orelse 1.0);
            const min_float: f64 = @floatCast(min_value orelse 0.0);

            const diff = max_float - min_float;
            break :blk diff * diff / 12.0;
        },
        else => unreachable,
    };

    try std.testing.expectApproxEqRel(expected_variance, variance, 0.1);
}

fn test_random_fill(
    allocator: std.mem.Allocator,
    ctx: *wekua.core.Context,
    comptime T: type,
    comptime is_complex: bool,
    comptime vectors_enabled: bool,

    min_value: ?T,
    max_value: ?T,
) !void {
    const command_queue = &ctx.command_queues[0];

    const tensor = try wekua.Tensor(T).alloc(ctx, &.{ 100, 100, 100 }, .{
        .is_complex = is_complex,
        .vectors_enabled = vectors_enabled,
    });
    defer tensor.release();

    if (command_queue.typeIsSupported(T)) {
        try wekua.tensor.random.uniform(T, command_queue, tensor, null, min_value, max_value);
        try checkVariance(T, allocator, command_queue, tensor, min_value, max_value);

        const seed = std.crypto.random.int(u64);
        try wekua.tensor.random.uniform(T, command_queue, tensor, seed, min_value, max_value);
        try checkVariance(T, allocator, command_queue, tensor, min_value, max_value);
    }
}

test "Check variance" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    inline for (wekua.core.SupportedTypes) |T| {
        try test_random_fill(allocator, ctx, T, false, true, null, null);
        try test_random_fill(allocator, ctx, T, false, false, null, null);

        try test_random_fill(allocator, ctx, T, true, true, null, null);
        try test_random_fill(allocator, ctx, T, true, false, null, null);
    }
}

test "Check variance with range" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    inline for (wekua.core.SupportedTypes) |T| {
        for (0..20) |_| {
            const min_value: T = switch (@typeInfo(T)) {
                .int => std.crypto.random.intRangeAtMost(T, 1, std.math.maxInt(T)/4),
                .float => -2000 + std.crypto.random.float(T) * 4000,
                else => unreachable,
            };

            const max_value: T = switch (@typeInfo(T)) {
                .int => std.crypto.random.intRangeAtMost(T, 3*std.math.maxInt(T)/4, std.math.maxInt(T)),
                .float => min_value + std.crypto.random.float(T) * 4000,
                else => unreachable,
            };

            try test_random_fill(allocator, ctx, T, false, true, min_value, max_value);
            try test_random_fill(allocator, ctx, T, false, false, min_value, max_value);

            try test_random_fill(allocator, ctx, T, true, true, min_value, max_value);
            try test_random_fill(allocator, ctx, T, true, false, min_value, max_value);
        }
    }
}

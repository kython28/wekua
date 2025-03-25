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
            try wekua.tensor.Random.fill(T, tensor, command_queue, null);

            const seed = std.crypto.random.int(u64);
            try wekua.tensor.Random.fill(T, tensor, command_queue, seed);
        }
    }
}

fn checkVariance(
    comptime T: type,
    allocator: std.mem.Allocator,
    command_queue: *const wekua.core.CommandQueue,
    tensor: *wekua.Tensor(T),
) !void {
    const buf = try allocator.alloc(T, tensor.number_of_elements_without_padding);
    defer allocator.free(buf);

    try wekua.tensor.memory.writeToBuffer(T, tensor, command_queue, buf);

    var mean: f64 = 0;
    for (buf) |elem| {
        const val: f64 = switch (@typeInfo(T)) {
            .int => @floatFromInt(elem),
            .float => @floatCast(elem),
            else => unreachable
        };

        mean += val;
    }

    const number_of_elements_float: f64 = @floatFromInt(tensor.number_of_elements_without_padding);
    mean /= number_of_elements_float;

    var sq_diff: f64 = 0.0;
    for (buf) |elem| {
        const diff = switch(@typeInfo(T)) {
            .int => @as(f64, @floatFromInt(elem)) - mean,
            .float => @as(f64, @floatCast(elem)) - mean,
            else => unreachable
        };
        sq_diff += diff * diff;
    }

    const variance: f64 = sq_diff / (number_of_elements_float - 1);


    const expected_variance = switch (@typeInfo(T)) {
        .int => blk: {
            const max_int: f64 = @floatFromInt(std.math.maxInt(T));
            const min_int: f64 = @floatFromInt(std.math.minInt(T));

            break :blk (max_int - min_int) * (max_int - min_int) / 12.0;
        },
        .float => 1.0 / 12.0,
        else => unreachable
    };
    const epsilon = variance*1.96/std.math.sqrt(number_of_elements_float);
    try std.testing.expectApproxEqAbs(expected_variance, variance, epsilon);
}

test "Check variance" {
    const allocator = std.testing.allocator;

    const ctx = try wekua.core.Context.init_from_device_type(allocator, null, cl.device.enums.device_type.all);
    defer ctx.release();

    const command_queue = &ctx.command_queues[0];

    inline for (wekua.core.SupportedTypes) |T| {
        const tensor = try wekua.Tensor(T).alloc(ctx, &.{ 20, 20, 20 }, .{});
        defer tensor.release();

        if (command_queue.typeIsSupported(T)) {
            try wekua.tensor.Random.fill(T, tensor, command_queue, null);
            try checkVariance(T, allocator, command_queue, tensor);

            try wekua.tensor.Random.fill(T, tensor, command_queue, 0);
            try checkVariance(T, allocator, command_queue, tensor);
        }
    }
}

pub const getValue = @import("get_value.zig").getValue;
pub const putValue = @import("put_value.zig").putValue;
// pub const readFromBuffer = @import("read_from_buffer.zig").readFromBuffer;
// pub const writeToBuffer = @import("write_to_buffer.zig").writeToBuffer;
// pub const copy = @import("copy.zig").copy;

// -----------------------------------------------------------------------------
// Unit Tests
const std = @import("std");
const testing = std.testing;
const cl = @import("opencl");

const core = @import("core");
const Context = core.Context;
const Pipeline = core.Pipeline;

const tensor_module = @import("../main.zig");
const Tensor = tensor_module.Tensor;

test "putValue and getValue - 1D tensor for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const expected: T = blk: {
                if (comptime core.types.isComplex(T)) {
                    break :blk .{ .real = 42, .imag = 24 };
                } else {
                    break :blk 42;
                }
            };

            const coor = [_]u64{7};

            // Write value
            try putValue(T, pipeline, tensor, &coor, &expected);

            // Read value back
            var result: T = undefined;
            try getValue(T, pipeline, tensor, &coor, &result);
            pipeline.waitAndCleanup();

            // Verify
            if (comptime core.types.isComplex(T)) {
                try testing.expectEqual(expected.real, result.real);
                try testing.expectEqual(expected.imag, result.imag);
            } else {
                try testing.expectEqual(expected, result);
            }
        }
    }
}

test "putValue and getValue - 2D tensor for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const expected1: T = if (comptime core.types.isComplex(T))
                .{ .real = 10, .imag = 20 }
            else
                10;
            const expected2: T = if (comptime core.types.isComplex(T))
                .{ .real = 30, .imag = 40 }
            else
                30;

            const coor1 = [_]u64{ 0, 0 };
            const coor2 = [_]u64{ 2, 3 };

            // Write values
            try putValue(T, pipeline, tensor, &coor1, &expected1);
            try putValue(T, pipeline, tensor, &coor2, &expected2);

            // Read values back
            var result1: T = undefined;
            var result2: T = undefined;
            try getValue(T, pipeline, tensor, &coor1, &result1);
            try getValue(T, pipeline, tensor, &coor2, &result2);
            pipeline.waitAndCleanup();

            // Verify
            if (comptime core.types.isComplex(T)) {
                try testing.expectEqual(expected1.real, result1.real);
                try testing.expectEqual(expected1.imag, result1.imag);
                try testing.expectEqual(expected2.real, result2.real);
                try testing.expectEqual(expected2.imag, result2.imag);
            } else {
                try testing.expectEqual(expected1, result1);
                try testing.expectEqual(expected2, result2);
            }
        }
    }
}

test "putValue and getValue - 3D tensor for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const expected: T = if (comptime core.types.isComplex(T))
                .{ .real = 99, .imag = 88 }
            else
                99;

            const coor = [_]u64{ 1, 2, 3 };

            // Write value
            try putValue(T, pipeline, tensor, &coor, &expected);

            // Read value back
            var result: T = undefined;
            try getValue(T, pipeline, tensor, &coor, &result);
            pipeline.waitAndCleanup();

            // Verify
            if (comptime core.types.isComplex(T)) {
                try testing.expectEqual(expected.real, result.real);
                try testing.expectEqual(expected.imag, result.imag);
            } else {
                try testing.expectEqual(expected, result);
            }
        }
    }
}

test "putValue and getValue - multiple values same tensor for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 5, 5 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Write multiple values
            const coords = [_][2]u64{
                .{ 0, 0 },
                .{ 1, 1 },
                .{ 2, 2 },
                .{ 3, 3 },
                .{ 4, 4 },
            };

            for (coords, 0..) |coor, i| {
                var value: T = undefined;
                if (comptime core.types.isComplex(T)) {
                    value = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i * 10), .imag = @floatFromInt(i * 5) },
                        .int => .{ .real = @intCast(i * 10), .imag = @intCast(i * 5) },
                        else => unreachable
                    };
                }else{
                    value = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 10),
                        .int => @intCast(i * 10),
                        else => unreachable
                    };
                }

                try putValue(T, pipeline, tensor, &coor, &value);
                pipeline.waitAndCleanup();
            }

            // Read and verify all values
            for (coords, 0..) |coor, i| {
                var expected: T = undefined;
                if (comptime core.types.isComplex(T)) {
                    expected = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i * 10), .imag = @floatFromInt(i * 5) },
                        .int => .{ .real = @intCast(i * 10), .imag = @intCast(i * 5) },
                        else => unreachable
                    };
                }else{
                    expected = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 10),
                        .int => @intCast(i * 10),
                        else => unreachable
                    };
                }

                var result: T = undefined;
                try getValue(T, pipeline, tensor, &coor, &result);
                pipeline.waitAndCleanup();

                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result.real);
                    try testing.expectEqual(expected.imag, result.imag);
                } else {
                    try testing.expectEqual(expected, result);
                }
            }
        }
    }
}

test "getValue - read zeroed tensor for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const coor = [_]u64{ 1, 1 };

            // Read value without writing (should be zero from alloc)
            var result: T = undefined;
            try getValue(T, pipeline, tensor, &coor, &result);
            pipeline.waitAndCleanup();

            // Verify it's zero
            if (comptime core.types.isComplex(T)) {
                try testing.expectEqual(@as(@TypeOf(result.real), 0), result.real);
                try testing.expectEqual(@as(@TypeOf(result.imag), 0), result.imag);
            } else {
                try testing.expectEqual(@as(T, 0), result);
            }
        }
    }
}

test "putValue and getValue - overwrite values for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const coor = [_]u64{2};

            // Write first value
            const value1: T = if (comptime core.types.isComplex(T))
                .{ .real = 10, .imag = 20 }
            else
                10;
            try putValue(T, pipeline, tensor, &coor, &value1);

            // Write second value (overwrite)
            const value2: T = if (comptime core.types.isComplex(T))
                .{ .real = 99, .imag = 88 }
            else
                99;
            try putValue(T, pipeline, tensor, &coor, &value2);

            // Read value - should be the second one
            var result: T = undefined;
            try getValue(T, pipeline, tensor, &coor, &result);
            pipeline.waitAndCleanup();

            // Verify it's the second value
            if (comptime core.types.isComplex(T)) {
                try testing.expectEqual(value2.real, result.real);
                try testing.expectEqual(value2.imag, result.imag);
            } else {
                try testing.expectEqual(value2, result);
            }
        }
    }
}

test "putValue - invalid coordinates dimension mismatch" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    const value: f32 = 42.0;
    const coor = [_]u64{0}; // Wrong dimension

    const result = putValue(f32, pipeline, tensor, &coor, &value);
    try testing.expectError(tensor_module.Errors.InvalidCoordinates, result);
}

test "putValue - invalid coordinates out of bounds" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    const value: f32 = 42.0;
    const coor = [_]u64{ 0, 5 }; // Out of bounds

    const result = putValue(f32, pipeline, tensor, &coor, &value);
    try testing.expectError(tensor_module.Errors.InvalidCoordinates, result);
}

test "getValue - invalid coordinates dimension mismatch" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    var result: f32 = undefined;
    const coor = [_]u64{0}; // Wrong dimension

    const err = getValue(f32, pipeline, tensor, &coor, &result);
    try testing.expectError(tensor_module.Errors.InvalidCoordinates, err);
}

test "getValue - invalid coordinates out of bounds" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    var result: f32 = undefined;
    const coor = [_]u64{ 0, 5 }; // Out of bounds

    const err = getValue(f32, pipeline, tensor, &coor, &result);
    try testing.expectError(tensor_module.Errors.InvalidCoordinates, err);
}

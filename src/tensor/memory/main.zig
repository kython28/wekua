pub const getValue = @import("get_value.zig").getValue;
pub const putValue = @import("put_value.zig").putValue;
pub const readFromBuffer = @import("read_from_buffer.zig").readFromBuffer;
pub const writeToBuffer = @import("write_to_buffer.zig").writeToBuffer;
pub const copy = @import("copy.zig").copy;

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

test "readFromBuffer and writeToBuffer - 1D tensor for all types" {
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
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const buffer_size = shape[0];

            // Create input buffer with known values
            const input_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_buffer);

            for (input_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i), .imag = @floatFromInt(i + 100) },
                        .int => .{ .real = @intCast(i), .imag = @intCast(i + 100) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i),
                        .int => @intCast(i),
                        else => unreachable,
                    };
                }
            }

            // Write buffer to tensor
            try readFromBuffer(T, pipeline, tensor, input_buffer);

            // Read tensor to output buffer
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // Verify buffers match
            for (input_buffer, output_buffer) |expected, result_val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result_val.real);
                    try testing.expectEqual(expected.imag, result_val.imag);
                } else {
                    try testing.expectEqual(expected, result_val);
                }
            }
        }
    }
}

test "readFromBuffer and writeToBuffer - 2D tensor for all types" {
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
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const buffer_size = shape[0] * shape[1];

            // Create input buffer
            const input_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_buffer);

            for (input_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i * 2), .imag = @floatFromInt(i * 3) },
                        .int => .{ .real = @intCast(i * 2), .imag = @intCast(i * 3) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 2),
                        .int => @intCast(i * 2),
                        else => unreachable,
                    };
                }
            }

            // Write to tensor
            try readFromBuffer(T, pipeline, tensor, input_buffer);

            // Read from tensor
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // Verify
            for (input_buffer, output_buffer) |expected, result_val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result_val.real);
                    try testing.expectEqual(expected.imag, result_val.imag);
                } else {
                    try testing.expectEqual(expected, result_val);
                }
            }
        }
    }
}

test "readFromBuffer and writeToBuffer - 3D tensor for all types" {
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
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const buffer_size = shape[0] * shape[1] * shape[2];

            // Create input buffer
            const input_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_buffer);

            for (input_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i), .imag = @floatFromInt(buffer_size - i) },
                        .int => .{ .real = @intCast(i), .imag = @intCast(buffer_size - i) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i),
                        .int => @intCast(i),
                        else => unreachable,
                    };
                }
            }

            // Write to tensor
            try readFromBuffer(T, pipeline, tensor, input_buffer);

            // Read from tensor
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // Verify
            for (input_buffer, output_buffer) |expected, result_val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result_val.real);
                    try testing.expectEqual(expected.imag, result_val.imag);
                } else {
                    try testing.expectEqual(expected, result_val);
                }
            }
        }
    }
}

test "readFromBuffer - invalid buffer size" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).empty(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    // Create buffer with wrong size
    const wrong_buffer = try allocator.alloc(f32, 5); // Should be 6
    defer allocator.free(wrong_buffer);

    const err = readFromBuffer(f32, pipeline, tensor, wrong_buffer);
    try testing.expectError(tensor_module.Errors.InvalidBuffer, err);
}

test "writeToBuffer - invalid buffer size" {
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

    // Create buffer with wrong size
    const wrong_buffer = try allocator.alloc(f32, 10); // Should be 6
    defer allocator.free(wrong_buffer);

    const err = writeToBuffer(f32, pipeline, tensor, wrong_buffer);
    try testing.expectError(tensor_module.Errors.InvalidBuffer, err);
}

test "readFromBuffer and writeToBuffer - round trip preserves data" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 4, 5 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).empty(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const buffer_size = shape[0] * shape[1];

            // Create buffers
            const buffer1 = try allocator.alloc(T, buffer_size);
            defer allocator.free(buffer1);
            const buffer2 = try allocator.alloc(T, buffer_size);
            defer allocator.free(buffer2);
            const buffer3 = try allocator.alloc(T, buffer_size);
            defer allocator.free(buffer3);

            // Initialize first buffer
            for (buffer1, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i * 7), .imag = @floatFromInt(i * 11) },
                        .int => .{ .real = @intCast(i * 2), .imag = @intCast(i) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 7),
                        .int => @intCast(i * 2),
                        else => unreachable,
                    };
                }
            }

            // First round trip: buffer1 -> tensor -> buffer2
            try readFromBuffer(T, pipeline, tensor, buffer1);

            try writeToBuffer(T, pipeline, tensor, buffer2);

            // Second round trip: buffer2 -> tensor -> buffer3
            try readFromBuffer(T, pipeline, tensor, buffer2);

            try writeToBuffer(T, pipeline, tensor, buffer3);
            pipeline.waitAndCleanup();

            // All buffers should match
            for (buffer1, buffer2, buffer3) |val1, val2, val3| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(val1.real, val2.real);
                    try testing.expectEqual(val1.imag, val2.imag);
                    try testing.expectEqual(val2.real, val3.real);
                    try testing.expectEqual(val2.imag, val3.imag);
                } else {
                    try testing.expectEqual(val1, val2);
                    try testing.expectEqual(val2, val3);
                }
            }
        }
    }
}

test "copy - same row_pitch for all types" {
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
            // Create two tensors with same configuration (same row_pitch)
            const src = try Tensor(T).empty(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const dst = try Tensor(T).empty(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Verify they have the same row_pitch
            try testing.expectEqual(src.memory_layout.row_pitch, dst.memory_layout.row_pitch);

            const buffer_size = shape[0] * shape[1];

            // Fill source tensor with data
            const src_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(src_buffer);

            for (src_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i * 3), .imag = @floatFromInt(i * 5) },
                        .int => .{ .real = @intCast(i * 3), .imag = @intCast(i * 5) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 3),
                        .int => @intCast(i * 3),
                        else => unreachable,
                    };
                }
            }

            try readFromBuffer(T, pipeline, src, src_buffer);

            // Copy from src to dst
            try copy(T, pipeline, src, dst);

            // Read dst back and verify
            const dst_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(dst_buffer);

            try writeToBuffer(T, pipeline, dst, dst_buffer);
            pipeline.waitAndCleanup();

            // Verify both buffers are equal
            for (src_buffer, dst_buffer) |expected, result_val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result_val.real);
                    try testing.expectEqual(expected.imag, result_val.imag);
                } else {
                    try testing.expectEqual(expected, result_val);
                }
            }
        }
    }
}

test "copy - different row_pitch for all types" {
    const allocator = testing.allocator;

    const context = try Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };

    inline for (core.types.SupportedTypes) |T| {
        if (command_queue.isTypeSupported(T) and !core.types.isComplex(T)) {
            // Create source tensor with vectors enabled
            const config_with_vectors = tensor_module.CreateConfig{ .vectors_enabled = true };
            const src = try Tensor(T).empty(context, pipeline, &shape, config_with_vectors);
            defer src.release(pipeline);

            // Create destination tensor with vectors disabled
            const config_without_vectors = tensor_module.CreateConfig{ .vectors_enabled = false };
            const dst = try Tensor(T).empty(context, pipeline, &shape, config_without_vectors);
            defer dst.release(pipeline);

            // Verify they have different row_pitch (if vectors are actually used)
            if (src.flags.vectors_enabled and !dst.flags.vectors_enabled) {
                // Only test if vectors are actually different
                const buffer_size = shape[0] * shape[1];

                // Fill source tensor with data
                const src_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(src_buffer);

                for (src_buffer, 0..) |*val, i| {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 7),
                        .int => @intCast(i * 7),
                        else => unreachable,
                    };
                }

                try readFromBuffer(T, pipeline, src, src_buffer);

                // Copy from src to dst (different row_pitch)
                try copy(T, pipeline, src, dst);

                // Read dst back and verify
                const dst_buffer = try allocator.alloc(T, buffer_size);
                defer allocator.free(dst_buffer);

                try writeToBuffer(T, pipeline, dst, dst_buffer);
                pipeline.waitAndCleanup();

                // Verify both buffers are equal
                for (src_buffer, dst_buffer) |expected, result_val| {
                    try testing.expectEqual(expected, result_val);
                }
            }
        }
    }
}

test "copy - 1D tensor for all types" {
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
            const src = try Tensor(T).empty(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const dst = try Tensor(T).empty(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            const buffer_size = shape[0];

            // Fill source
            const src_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(src_buffer);

            for (src_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i), .imag = @floatFromInt(i + 50) },
                        .int => .{ .real = @intCast(i), .imag = @intCast(i + 50) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i),
                        .int => @intCast(i),
                        else => unreachable,
                    };
                }
            }

            try readFromBuffer(T, pipeline, src, src_buffer);

            // Copy
            try copy(T, pipeline, src, dst);

            // Verify
            const dst_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(dst_buffer);

            try writeToBuffer(T, pipeline, dst, dst_buffer);
            pipeline.waitAndCleanup();

            for (src_buffer, dst_buffer) |expected, result_val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result_val.real);
                    try testing.expectEqual(expected.imag, result_val.imag);
                } else {
                    try testing.expectEqual(expected, result_val);
                }
            }
        }
    }
}

test "copy - 3D tensor for all types" {
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
            const src = try Tensor(T).empty(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const dst = try Tensor(T).empty(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            const buffer_size = shape[0] * shape[1] * shape[2];

            // Fill source
            const src_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(src_buffer);

            for (src_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i * 2), .imag = @floatFromInt(buffer_size - i) },
                        .int => .{ .real = @intCast(i * 2), .imag = @intCast(buffer_size - @as(usize, @intCast(i))) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i * 2),
                        .int => @intCast(i * 2),
                        else => unreachable,
                    };
                }
            }

            try readFromBuffer(T, pipeline, src, src_buffer);

            // Copy
            try copy(T, pipeline, src, dst);

            // Verify
            const dst_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(dst_buffer);

            try writeToBuffer(T, pipeline, dst, dst_buffer);
            pipeline.waitAndCleanup();

            for (src_buffer, dst_buffer) |expected, result_val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, result_val.real);
                    try testing.expectEqual(expected.imag, result_val.imag);
                } else {
                    try testing.expectEqual(expected, result_val);
                }
            }
        }
    }
}

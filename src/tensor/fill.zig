const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const helpers = @import("helpers.zig");

const tensor_module = @import("main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const fill_cl_kernel: []const u8 = @embedFile("kernels/fill.cl");

pub fn constant(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
    scalar: T,
) TensorErrors!void {
    const command_queue = pipeline.command_queue;
    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        .Fill,
        "fill",
        fill_cl_kernel,
        null,
    );

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try setArg(kernel, 1, @sizeOf(u64), @ptrCast(&tensor.memory_layout.row_pitch));
    try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.memory_layout.slice_pitch));
    try setArg(kernel, 3, @sizeOf(T), @ptrCast(&scalar));


    // TODO: Adapt code to use views
    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        pipeline.command_queue.cl_command_queue,
        kernel,
        null,
        &tensor.work_configuration.global_work_items_without_vectors,
        &tensor.work_configuration.local_work_items_without_vectors[pipeline.command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

pub inline fn one(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) !void {
    try constant(T, pipeline, tensor, switch (comptime core.types.isComplex(T)) {
        true => .{ .real = 1, .imag = 0 },
        false => 1,
    });
}

pub fn zeroes(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    const prev_events = pipeline.prevEvents();

    const zero: T = std.mem.zeroes(T);

    var new_event: cl.event.Event = undefined;
    try cl.buffer.fill(
        pipeline.command_queue.cl_command_queue,
        tensor.buffer,
        &zero,
        @sizeOf(T),
        0,
        tensor.memory_layout.size,
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

// -----------------------------------------------------------------------------
// Unit Tests
const testing = std.testing;

const memory = @import("memory/main.zig");

test "constant - 1D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const expected: T = if (comptime core.types.isComplex(T))
                .{ .real = 42, .imag = 24 }
            else
                42;

            // Fill with constant
            try constant(T, pipeline, tensor, expected);

            // Read back and verify all values
            const buffer_size = shape[0];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should equal expected
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, val.real);
                    try testing.expectEqual(expected.imag, val.imag);
                } else {
                    try testing.expectEqual(expected, val);
                }
            }
        }
    }
}

test "constant - 2D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const expected: T = if (comptime core.types.isComplex(T))
                .{ .real = 99, .imag = 88 }
            else
                99;

            // Fill with constant
            try constant(T, pipeline, tensor, expected);

            // Read back and verify
            const buffer_size = shape[0] * shape[1];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should equal expected
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, val.real);
                    try testing.expectEqual(expected.imag, val.imag);
                } else {
                    try testing.expectEqual(expected, val);
                }
            }
        }
    }
}

test "constant - 3D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const expected: T = if (comptime core.types.isComplex(T))
                .{ .real = 7, .imag = 13 }
            else
                7;

            // Fill with constant
            try constant(T, pipeline, tensor, expected);

            // Read back and verify
            const buffer_size = shape[0] * shape[1] * shape[2];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should equal expected
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(expected.real, val.real);
                    try testing.expectEqual(expected.imag, val.imag);
                } else {
                    try testing.expectEqual(expected, val);
                }
            }
        }
    }
}

test "zeroes - 1D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Fill with zeroes
            try zeroes(T, pipeline, tensor);

            // Read back and verify all values are zero
            const buffer_size = shape[0];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should be zero
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(val.real), 0), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 0), val);
                }
            }
        }
    }
}

test "zeroes - 2D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Fill with zeroes
            try zeroes(T, pipeline, tensor);

            // Read back and verify
            const buffer_size = shape[0] * shape[1];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should be zero
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(val.real), 0), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 0), val);
                }
            }
        }
    }
}

test "zeroes - 3D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Fill with zeroes
            try zeroes(T, pipeline, tensor);

            // Read back and verify
            const buffer_size = shape[0] * shape[1] * shape[2];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should be zero
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(val.real), 0), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 0), val);
                }
            }
        }
    }
}

test "zeroes - overwrites previous data for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // First fill with a constant value
            const value: T = if (comptime core.types.isComplex(T))
                .{ .real = 99, .imag = 88 }
            else
                99;

            try constant(T, pipeline, tensor, value);

            // Then fill with zeroes
            try zeroes(T, pipeline, tensor);

            // Read back and verify all values are zero
            const buffer_size = shape[0] * shape[1];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should be zero (not the previous value)
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(val.real), 0), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 0), val);
                }
            }
        }
    }
}

test "constant - different values for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Fill with first value
            const value1: T = if (comptime core.types.isComplex(T))
                .{ .real = 10, .imag = 20 }
            else
                10;

            try constant(T, pipeline, tensor, value1);

            // Read and verify
            const buffer_size = shape[0];
            const output_buffer1 = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer1);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer1);
            pipeline.waitAndCleanup();

            for (output_buffer1) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(value1.real, val.real);
                    try testing.expectEqual(value1.imag, val.imag);
                } else {
                    try testing.expectEqual(value1, val);
                }
            }

            // Fill with second value (overwrite)
            const value2: T = if (comptime core.types.isComplex(T))
                .{ .real = 77, .imag = 66 }
            else
                77;

            try constant(T, pipeline, tensor, value2);

            // Read and verify
            const output_buffer2 = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer2);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer2);
            pipeline.waitAndCleanup();

            for (output_buffer2) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(value2.real, val.real);
                    try testing.expectEqual(value2.imag, val.imag);
                } else {
                    try testing.expectEqual(value2, val);
                }
            }
        }
    }
}

test "one - 1D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Fill with ones
            try one(T, pipeline, tensor);

            // Read back and verify all values are 1
            const buffer_size = shape[0];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should be 1 (or 1+0i for complex)
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(val.real), 1), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 1), val);
                }
            }
        }
    }
}

test "one - 2D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Fill with ones
            try one(T, pipeline, tensor);

            // Read back and verify
            const buffer_size = shape[0] * shape[1];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should be 1 (or 1+0i for complex)
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(val.real), 1), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 1), val);
                }
            }
        }
    }
}

test "one - 3D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Fill with ones
            try one(T, pipeline, tensor);

            // Read back and verify
            const buffer_size = shape[0] * shape[1] * shape[2];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // All values should be 1 (or 1+0i for complex)
            for (output_buffer) |val| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(val.real), 1), val.real);
                    try testing.expectEqual(@as(@TypeOf(val.imag), 0), val.imag);
                } else {
                    try testing.expectEqual(@as(T, 1), val);
                }
            }
        }
    }
}

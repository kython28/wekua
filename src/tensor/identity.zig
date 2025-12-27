const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const utils = @import("utils");
const helpers = @import("helpers.zig");

const identity_cl_kernel: []const u8 = @embedFile("kernels/identity.cl");

pub fn identity(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    const size = tensor.dimensions.shape[0];
    for (tensor.dimensions.shape[1..]) |s| {
        if (s != size) {
            return tensor_module.Errors.InvalidValue;
        }
    }

    try tensor_module.fill.zeroes(T, pipeline, tensor);

    const command_queue = pipeline.command_queue;
    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        .Identity,
        "identity",
        identity_cl_kernel,
        null,
    );
    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    var work_items: u64 = undefined;
    utils.calculateWorkItems(
        &.{ size },
        @as([*]u64, @ptrCast(&work_items))[0..1],
        command_queue.max_work_group_size,
    );

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&tensor.pitches_buffer));
    try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&tensor.dimensions.shape.len));

    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        &.{ size },
        &.{ work_items },
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

// -----------------------------------------------------------------------------
// Unit Tests
const std = @import("std");
const testing = std.testing;

const memory = tensor_module.memory;

test "identity - 2D square matrix for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Apply identity
            try identity(T, pipeline, tensor);

            // Verify diagonal elements are 1 and others are 0
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    const coords = [_]u64{ i, j };
                    var value: T = undefined;

                    try memory.getValue(T, pipeline, tensor, &coords, &value);
                    pipeline.waitAndCleanup();

                    if (i == j) {
                        // Diagonal should be 1
                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(@as(@TypeOf(value.real), 1), value.real);
                            try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
                        } else {
                            try testing.expectEqual(@as(T, 1), value);
                        }
                    } else {
                        // Off-diagonal should be 0
                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(@as(@TypeOf(value.real), 0), value.real);
                            try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
                        } else {
                            try testing.expectEqual(@as(T, 0), value);
                        }
                    }
                }
            }
        }
    }
}

test "identity - 3D cubic tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 4, 4, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Apply identity
            try identity(T, pipeline, tensor);

            // Verify diagonal elements (i,i,i) are 1 and others are 0
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    for (0..shape[2]) |k| {
                        const coords = [_]u64{ i, j, k };
                        var value: T = undefined;

                        try memory.getValue(T, pipeline, tensor, &coords, &value);
                        pipeline.waitAndCleanup();

                        if (i == j and j == k) {
                            // Diagonal should be 1
                            if (comptime core.types.isComplex(T)) {
                                try testing.expectEqual(@as(@TypeOf(value.real), 1), value.real);
                                try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
                            } else {
                                try testing.expectEqual(@as(T, 1), value);
                            }
                        } else {
                            // Off-diagonal should be 0
                            if (comptime core.types.isComplex(T)) {
                                try testing.expectEqual(@as(@TypeOf(value.real), 0), value.real);
                                try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
                            } else {
                                try testing.expectEqual(@as(T, 0), value);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "identity - 4D hypercubic tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 3, 3, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Apply identity
            try identity(T, pipeline, tensor);

            // Verify diagonal elements (i,i,i,i) are 1
            // Sample a few diagonal and off-diagonal elements
            for (0..3) |i| {
                const diag_coords = [_]u64{ i, i, i, i };
                var diag_value: T = undefined;

                try memory.getValue(T, pipeline, tensor, &diag_coords, &diag_value);
                pipeline.waitAndCleanup();

                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(diag_value.real), 1), diag_value.real);
                    try testing.expectEqual(@as(@TypeOf(diag_value.imag), 0), diag_value.imag);
                } else {
                    try testing.expectEqual(@as(T, 1), diag_value);
                }

                // Check an off-diagonal element
                const off_diag_coords = [_]u64{ i, i, i, (i + 1) % 3 };
                var off_diag_value: T = undefined;

                try memory.getValue(T, pipeline, tensor, &off_diag_coords, &off_diag_value);
                pipeline.waitAndCleanup();

                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(@as(@TypeOf(off_diag_value.real), 0), off_diag_value.real);
                    try testing.expectEqual(@as(@TypeOf(off_diag_value.imag), 0), off_diag_value.imag);
                } else {
                    try testing.expectEqual(@as(T, 0), off_diag_value);
                }
            }
        }
    }
}

test "identity - different sizes for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const sizes = [_]u64{ 2, 5, 8 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            for (sizes) |size| {
                const shape = [_]u64{ size, size };
                const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
                defer tensor.release(pipeline);

                // Apply identity
                try identity(T, pipeline, tensor);

                // Verify a few diagonal and off-diagonal elements
                const samples = @min(size, 3);
                for (0..samples) |i| {
                    // Check diagonal
                    const diag_coords = [_]u64{ i, i };
                    var diag_value: T = undefined;

                    try memory.getValue(T, pipeline, tensor, &diag_coords, &diag_value);
                    pipeline.waitAndCleanup();

                    if (comptime core.types.isComplex(T)) {
                        try testing.expectEqual(@as(@TypeOf(diag_value.real), 1), diag_value.real);
                        try testing.expectEqual(@as(@TypeOf(diag_value.imag), 0), diag_value.imag);
                    } else {
                        try testing.expectEqual(@as(T, 1), diag_value);
                    }

                    // Check off-diagonal
                    if (size > 1) {
                        const off_diag_coords = [_]u64{ i, (i + 1) % size };
                        var off_diag_value: T = undefined;

                        try memory.getValue(T, pipeline, tensor, &off_diag_coords, &off_diag_value);
                        pipeline.waitAndCleanup();

                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(@as(@TypeOf(off_diag_value.real), 0), off_diag_value.real);
                            try testing.expectEqual(@as(@TypeOf(off_diag_value.imag), 0), off_diag_value.imag);
                        } else {
                            try testing.expectEqual(@as(T, 0), off_diag_value);
                        }
                    }
                }
            }
        }
    }
}

test "identity - invalid non-square matrix" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    // Should fail because matrix is not square
    const err = identity(f32, pipeline, tensor);
    try testing.expectError(tensor_module.Errors.InvalidValue, err);
}

test "identity - invalid non-cubic 3D tensor" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 3, 4 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    // Should fail because not all dimensions are equal
    const err = identity(f32, pipeline, tensor);
    try testing.expectError(tensor_module.Errors.InvalidValue, err);
}

test "identity - overwrites existing values for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 4, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Fill tensor with non-zero values
            const buffer_size = shape[0] * shape[1];
            const input_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_buffer);

            for (input_buffer, 0..) |*val, i| {
                if (comptime core.types.isComplex(T)) {
                    val.* = switch (@typeInfo(core.types.getType(T))) {
                        .float => .{ .real = @floatFromInt(i + 10), .imag = @floatFromInt(i + 20) },
                        .int => .{ .real = @intCast(i + 10), .imag = @intCast(i + 20) },
                        else => unreachable,
                    };
                } else {
                    val.* = switch (@typeInfo(T)) {
                        .float => @floatFromInt(i + 10),
                        .int => @intCast(i + 10),
                        else => unreachable,
                    };
                }
            }

            try memory.readFromBuffer(T, pipeline, tensor, input_buffer);

            // Apply identity (should overwrite all values)
            try identity(T, pipeline, tensor);

            // Verify result is identity matrix
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    const coords = [_]u64{ i, j };
                    var value: T = undefined;

                    try memory.getValue(T, pipeline, tensor, &coords, &value);
                    pipeline.waitAndCleanup();

                    if (i == j) {
                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(@as(@TypeOf(value.real), 1), value.real);
                            try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
                        } else {
                            try testing.expectEqual(@as(T, 1), value);
                        }
                    } else {
                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(@as(@TypeOf(value.real), 0), value.real);
                            try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
                        } else {
                            try testing.expectEqual(@as(T, 0), value);
                        }
                    }
                }
            }
        }
    }
}

test "identity - 1x1 matrix for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 1, 1 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Apply identity
            try identity(T, pipeline, tensor);

            // Verify single element is 1
            const coords = [_]u64{ 0, 0 };
            var value: T = undefined;

            try memory.getValue(T, pipeline, tensor, &coords, &value);
            pipeline.waitAndCleanup();

            if (comptime core.types.isComplex(T)) {
                try testing.expectEqual(@as(@TypeOf(value.real), 1), value.real);
                try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
            } else {
                try testing.expectEqual(@as(T, 1), value);
            }
        }
    }
}

test "identity - using writeToBuffer verification for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 5, 5 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            // Apply identity
            try identity(T, pipeline, tensor);

            // Read entire buffer
            const buffer_size = shape[0] * shape[1];
            const output_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, output_buffer);
            pipeline.waitAndCleanup();

            // Verify all elements
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    const idx = i * shape[1] + j;
                    const value = output_buffer[idx];

                    if (i == j) {
                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(@as(@TypeOf(value.real), 1), value.real);
                            try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
                        } else {
                            try testing.expectEqual(@as(T, 1), value);
                        }
                    } else {
                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(@as(@TypeOf(value.real), 0), value.real);
                            try testing.expectEqual(@as(@TypeOf(value.imag), 0), value.imag);
                        } else {
                            try testing.expectEqual(@as(T, 0), value);
                        }
                    }
                }
            }
        }
    }
}

const cl = @import("opencl");

const core = @import("core");
const KernelsSet = core.KernelsSet;
const Pipeline = core.Pipeline;

const tensor_module = @import("main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const helpers = @import("helpers.zig");
const init = @import("init.zig");

const transpose_cl_kernel: []const u8 = @embedFile("kernels/transpose.cl");
const transpose_2d_cl_kernel: []const u8 = @embedFile("kernels/transpose_2d.cl");

pub fn transpose(
    comptime T: type,
    pipeline: *Pipeline,
    result_tensor: *Tensor(T),
    tensor: *Tensor(T),
    dim0: u64,
    dim1: u64,
) TensorErrors!void {
    const shape_a = result_tensor.dimensions.shape;
    const shape_b = tensor.dimensions.shape;
    if (shape_a.len != shape_b.len) {
        return tensor_module.Errors.UnqualTensorsDimension;
    } else if (dim0 >= shape_a.len or dim1 >= shape_a.len) {
        return tensor_module.Errors.InvalidValue;
    } else if (tensor.dimensions.number_of_elements_without_padding != result_tensor.dimensions.number_of_elements_without_padding) {
        return tensor_module.Errors.UnqualTensorsDimension;
    } else if (shape_a[dim0] != shape_b[dim1] or shape_a[dim1] != shape_b[dim0]) {
        return tensor_module.Errors.InvalidValue;
    }

    if (dim0 == dim1) {
        try tensor_module.memory.copy(T, pipeline, tensor, result_tensor);
        return;
    }

    const command_queue = pipeline.command_queue;
    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        .Transpose,
        "transpose",
        transpose_cl_kernel,
        null,
    );

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const u64_size = @sizeOf(u64);
    const cl_mem_size = @sizeOf(cl.buffer.Mem);
    const shape = tensor.dimensions.shape;
    const ndim: u64 = @intCast(shape.len);
    var dim0_: u64 = undefined;
    var dim1_: u64 = undefined;

    if (dim0 > dim1) {
        dim0_ = dim1;
        dim1_ = dim0;
    } else {
        dim0_ = dim0;
        dim1_ = dim1;
    }

    const row_pitch = tensor.memory_layout.row_pitch;

    const tensor_width = tensor.work_configuration.global_work_items_without_vectors[2];
    const tensor_height = tensor.work_configuration.global_work_items_without_vectors[1] * row_pitch;

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&tensor.pitches_buffer));

    try setArg(kernel, 2, cl_mem_size, @ptrCast(&result_tensor.buffer));
    try setArg(kernel, 3, cl_mem_size, @ptrCast(&result_tensor.pitches_buffer));

    try setArg(kernel, 4, u64_size, @ptrCast(&row_pitch));
    try setArg(kernel, 5, u64_size, @ptrCast(&tensor.memory_layout.slice_pitch));
    try setArg(kernel, 6, u64_size, @ptrCast(&tensor_height));
    try setArg(kernel, 7, u64_size, @ptrCast(&tensor_width));

    try setArg(kernel, 8, u64_size, @ptrCast(&dim0_));
    try setArg(kernel, 9, u64_size, @ptrCast(&dim1_));
    try setArg(kernel, 10, u64_size, @ptrCast(&ndim));

    const wekua_id = command_queue.wekua_id;

    // TODO: Adapt code to use views
    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        &[1]u64{tensor.dimensions.number_of_elements},
        tensor.work_configuration.local_work_items_1d[wekua_id .. wekua_id + 1],
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

pub fn transpose_2d_inplace(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    if (tensor.dimensions.shape.len != 2) {
        return tensor_module.Errors.InvalidValue;
    }

    var shape: [2]u64 = undefined;
    shape[0] = tensor.dimensions.shape[1];
    shape[1] = tensor.dimensions.shape[0];

    const command_queue = pipeline.command_queue;
    const kernel = try KernelsSet.getClNoVectorKernel(
        T,
        command_queue,
        .Transpose2D,
        "transpose_2d",
        transpose_2d_cl_kernel,
        null,
    );

    const prev_events = pipeline.prevEvents();

    const A_row_pitch = tensor.memory_layout.row_pitch;
    const AT_row_pitch = tensor.memory_layout.slice_pitch / A_row_pitch;

    const setArg = cl.kernel.setArg;
    const u64_size = @sizeOf(u64);
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));
    try setArg(kernel, 1, u64_size, @ptrCast(&A_row_pitch));
    try setArg(kernel, 2, u64_size, @ptrCast(&AT_row_pitch));

    const wekua_id = command_queue.wekua_id;
    const global_work_items: []const u64 = tensor.work_configuration.global_work_items_without_vectors[1..];
    const local_work_items: []const u64 = tensor.work_configuration.local_work_items_without_vectors[wekua_id][1..];

    var transpose_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        global_work_items,
        local_work_items,
        prev_events,
        &transpose_event,
    );
    errdefer helpers.releaseEvent(transpose_event);

    _ = tensor.arena.reset(.retain_capacity);
    const arena_allocator = tensor.arena.allocator();

    // TODO: See how to handle errors here without panicking
    // Maybe it looks dangerous to use arena_allocator.dupe() here
    // but theoretically it can fail only when the arena reset fails
    // Anyway i will try to fix it later
    tensor.dimensions.shape = arena_allocator.dupe(u64, &shape) catch {
        @panic("Out of memory while defining new shape");
    };

    const pitches = arena_allocator.alloc(u64, 2) catch {
        @panic("Out of memory while defining new pitches");
    };
    tensor.dimensions.pitches = pitches;

    _ = try init.initTensorProperties(
        T,
        core.types.isComplex(T),
        core.types.getTypeId(T),
        tensor.context.command_queues,
        arena_allocator,
        tensor,
        &shape,
        tensor.flags.vectors_enabled,
    );

    var update_pitches_event: cl.event.Event = undefined;
    try cl.buffer.write(
        pipeline.command_queue.cl_command_queue,
        tensor.pitches_buffer,
        false,
        0,
        pitches.len * @sizeOf(u64),
        pitches.ptr,
        prev_events,
        &update_pitches_event,
    );
    errdefer helpers.releaseEvent(update_pitches_event);

    try pipeline.append(&.{transpose_event, update_pitches_event});
}

// -----------------------------------------------------------------------------
// Unit Tests
const std = @import("std");
const testing = std.testing;

const memory = @import("memory/main.zig");
const random = @import("random/main.zig");
const fill = @import("fill.zig");

test "transpose - 2D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const transposed_shape = [_]u64{ 4, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const result = try Tensor(T).alloc(context, pipeline, &transposed_shape, config);
            defer result.release(pipeline);

            // Fill with random values
            try random.uniform(T, pipeline, tensor, 42, null, null);

            // Transpose dimensions 0 and 1
            try transpose(T, pipeline, result, tensor, 0, 1);

            // Verify some specific values
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    const original_coords = [_]u64{ i, j };
                    const transposed_coords = [_]u64{ j, i };

                    var original_val: T = undefined;
                    var transposed_val: T = undefined;

                    try memory.getValue(T, pipeline, tensor, &original_coords, &original_val);
                    try memory.getValue(T, pipeline, result, &transposed_coords, &transposed_val);
                    pipeline.waitAndCleanup();

                    if (comptime core.types.isComplex(T)) {
                        try testing.expectEqual(original_val.real, transposed_val.real);
                        try testing.expectEqual(original_val.imag, transposed_val.imag);
                    } else {
                        try testing.expectEqual(original_val, transposed_val);
                    }
                }
            }
        }
    }
}

test "transpose - 3D tensor transpose dims 0 and 1 for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const transposed_shape = [_]u64{ 3, 2, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const result = try Tensor(T).alloc(context, pipeline, &transposed_shape, config);
            defer result.release(pipeline);

            // Fill with random values
            try random.uniform(T, pipeline, tensor, 123, null, null);

            // Transpose dimensions 0 and 1
            try transpose(T, pipeline, result, tensor, 0, 1);

            // Verify correctness
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    for (0..shape[2]) |k| {
                        const original_coords = [_]u64{ i, j, k };
                        const transposed_coords = [_]u64{ j, i, k };

                        var original_val: T = undefined;
                        var transposed_val: T = undefined;

                        try memory.getValue(T, pipeline, tensor, &original_coords, &original_val);
                        try memory.getValue(T, pipeline, result, &transposed_coords, &transposed_val);
                        pipeline.waitAndCleanup();

                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(original_val.real, transposed_val.real);
                            try testing.expectEqual(original_val.imag, transposed_val.imag);
                        } else {
                            try testing.expectEqual(original_val, transposed_val);
                        }
                    }
                }
            }
        }
    }
}

test "transpose - 3D tensor transpose dims 0 and 2 for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const transposed_shape = [_]u64{ 4, 3, 2 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const result = try Tensor(T).alloc(context, pipeline, &transposed_shape, config);
            defer result.release(pipeline);

            // Fill with random values
            try random.uniform(T, pipeline, tensor, 456, null, null);

            // Transpose dimensions 0 and 2
            try transpose(T, pipeline, result, tensor, 0, 2);

            // Verify correctness
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    for (0..shape[2]) |k| {
                        const original_coords = [_]u64{ i, j, k };
                        const transposed_coords = [_]u64{ k, j, i };

                        var original_val: T = undefined;
                        var transposed_val: T = undefined;

                        try memory.getValue(T, pipeline, tensor, &original_coords, &original_val);
                        try memory.getValue(T, pipeline, result, &transposed_coords, &transposed_val);
                        pipeline.waitAndCleanup();

                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(original_val.real, transposed_val.real);
                            try testing.expectEqual(original_val.imag, transposed_val.imag);
                        } else {
                            try testing.expectEqual(original_val, transposed_val);
                        }
                    }
                }
            }
        }
    }
}

test "transpose - 3D tensor transpose dims 1 and 2 for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const transposed_shape = [_]u64{ 2, 4, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const result = try Tensor(T).alloc(context, pipeline, &transposed_shape, config);
            defer result.release(pipeline);

            // Fill with random values
            try random.uniform(T, pipeline, tensor, 789, null, null);

            // Transpose dimensions 1 and 2
            try transpose(T, pipeline, result, tensor, 1, 2);

            // Verify correctness
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    for (0..shape[2]) |k| {
                        const original_coords = [_]u64{ i, j, k };
                        const transposed_coords = [_]u64{ i, k, j };

                        var original_val: T = undefined;
                        var transposed_val: T = undefined;

                        try memory.getValue(T, pipeline, tensor, &original_coords, &original_val);
                        try memory.getValue(T, pipeline, result, &transposed_coords, &transposed_val);
                        pipeline.waitAndCleanup();

                        if (comptime core.types.isComplex(T)) {
                            try testing.expectEqual(original_val.real, transposed_val.real);
                            try testing.expectEqual(original_val.imag, transposed_val.imag);
                        } else {
                            try testing.expectEqual(original_val, transposed_val);
                        }
                    }
                }
            }
        }
    }
}

test "transpose - 4D tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4, 5 };
    const transposed_shape = [_]u64{ 2, 5, 4, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const result = try Tensor(T).alloc(context, pipeline, &transposed_shape, config);
            defer result.release(pipeline);

            // Fill with random values
            try random.uniform(T, pipeline, tensor, 999, null, null);

            // Transpose dimensions 1 and 3
            try transpose(T, pipeline, result, tensor, 1, 3);

            // Verify a subset of values
            for (0..2) |i| {
                for (0..2) |j| {
                    for (0..2) |k| {
                        for (0..2) |l| {
                            const original_coords = [_]u64{ i, j, k, l };
                            const transposed_coords = [_]u64{ i, l, k, j };

                            var original_val: T = undefined;
                            var transposed_val: T = undefined;

                            try memory.getValue(T, pipeline, tensor, &original_coords, &original_val);
                            try memory.getValue(T, pipeline, result, &transposed_coords, &transposed_val);
                            pipeline.waitAndCleanup();

                            if (comptime core.types.isComplex(T)) {
                                try testing.expectEqual(original_val.real, transposed_val.real);
                                try testing.expectEqual(original_val.imag, transposed_val.imag);
                            } else {
                                try testing.expectEqual(original_val, transposed_val);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "transpose - same dimension copies tensor for all types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4, 5 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const result = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer result.release(pipeline);

            // Fill with random values
            try random.uniform(T, pipeline, tensor, 111, null, null);

            // Transpose same dimension (should copy)
            try transpose(T, pipeline, result, tensor, 1, 1);

            // Verify all values are copied
            const buffer_size = shape[0] * shape[1] * shape[2];
            const original_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(original_buffer);

            const result_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(result_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, original_buffer);
            try memory.writeToBuffer(T, pipeline, result, result_buffer);
            pipeline.waitAndCleanup();

            for (original_buffer, result_buffer) |orig, res| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(orig.real, res.real);
                    try testing.expectEqual(orig.imag, res.imag);
                } else {
                    try testing.expectEqual(orig, res);
                }
            }
        }
    }
}

test "transpose - dimension order does not matter" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const transposed_shape = [_]u64{ 4, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const result1 = try Tensor(T).alloc(context, pipeline, &transposed_shape, config);
            defer result1.release(pipeline);

            const result2 = try Tensor(T).alloc(context, pipeline, &transposed_shape, config);
            defer result2.release(pipeline);

            // Fill with random values
            try random.uniform(T, pipeline, tensor, 222, null, null);

            // Transpose with dim0=0, dim1=1
            try transpose(T, pipeline, result1, tensor, 0, 1);

            // Transpose with dim0=1, dim1=0 (reversed order)
            try transpose(T, pipeline, result2, tensor, 1, 0);

            // Both results should be identical
            const buffer_size = transposed_shape[0] * transposed_shape[1];
            const buffer1 = try allocator.alloc(T, buffer_size);
            defer allocator.free(buffer1);

            const buffer2 = try allocator.alloc(T, buffer_size);
            defer allocator.free(buffer2);

            try memory.writeToBuffer(T, pipeline, result1, buffer1);
            try memory.writeToBuffer(T, pipeline, result2, buffer2);
            pipeline.waitAndCleanup();

            for (buffer1, buffer2) |val1, val2| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(val1.real, val2.real);
                    try testing.expectEqual(val1.imag, val2.imag);
                } else {
                    try testing.expectEqual(val1, val2);
                }
            }
        }
    }
}

test "transpose - with known values" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const transposed_shape = [_]u64{ 3, 2 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    const result = try Tensor(f32).alloc(context, pipeline, &transposed_shape, config);
    defer result.release(pipeline);

    // Fill with known pattern: [[1, 2, 3], [4, 5, 6]]
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try memory.readFromBuffer(f32, pipeline, tensor, &input_data);

    // Transpose
    try transpose(f32, pipeline, result, tensor, 0, 1);

    // Read result
    const output_data = try allocator.alloc(f32, 6);
    defer allocator.free(output_data);

    try memory.writeToBuffer(f32, pipeline, result, output_data);
    pipeline.waitAndCleanup();

    // Expected: [[1, 4], [2, 5], [3, 6]]
    const expected = [_]f32{ 1, 4, 2, 5, 3, 6 };

    for (output_data, expected) |actual, exp| {
        try testing.expectEqual(exp, actual);
    }
}

test "transpose_2d_inplace - square matrix for all types" {
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

            // Save original values before transpose
            try random.uniform(T, pipeline, tensor, 42, null, null);

            var original_values: [3][3]T = undefined;
            for (0..3) |i| {
                for (0..3) |j| {
                    const coords = [_]u64{ i, j };
                    try memory.getValue(T, pipeline, tensor, &coords, &original_values[i][j]);
                    pipeline.waitAndCleanup();
                }
            }

            // Transpose in-place
            try transpose_2d_inplace(T, pipeline, tensor);

            // Shape should remain [3, 3]
            try testing.expectEqual(3, tensor.dimensions.shape[0]);
            try testing.expectEqual(3, tensor.dimensions.shape[1]);

            // Verify: transposed[j][i] == original[i][j]
            for (0..3) |i| {
                for (0..3) |j| {
                    const coords = [_]u64{ j, i };
                    var value: T = undefined;
                    try memory.getValue(T, pipeline, tensor, &coords, &value);
                    pipeline.waitAndCleanup();

                    if (comptime core.types.isComplex(T)) {
                        try testing.expectEqual(original_values[i][j].real, value.real);
                        try testing.expectEqual(original_values[i][j].imag, value.imag);
                    } else {
                        try testing.expectEqual(original_values[i][j], value);
                    }
                }
            }
        }
    }
}

test "transpose_2d_inplace - rectangular matrix for all types" {
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

            try random.uniform(T, pipeline, tensor, 123, null, null);

            // Save original values
            var original_values: [3][4]T = undefined;
            for (0..3) |i| {
                for (0..4) |j| {
                    const coords = [_]u64{ i, j };
                    try memory.getValue(T, pipeline, tensor, &coords, &original_values[i][j]);
                    pipeline.waitAndCleanup();
                }
            }

            // Transpose in-place
            try transpose_2d_inplace(T, pipeline, tensor);

            // Shape should be [4, 3]
            try testing.expectEqual(4, tensor.dimensions.shape[0]);
            try testing.expectEqual(3, tensor.dimensions.shape[1]);

            // Verify: transposed[j][i] == original[i][j]
            for (0..3) |i| {
                for (0..4) |j| {
                    const coords = [_]u64{ j, i };
                    var value: T = undefined;
                    try memory.getValue(T, pipeline, tensor, &coords, &value);
                    pipeline.waitAndCleanup();

                    if (comptime core.types.isComplex(T)) {
                        try testing.expectEqual(original_values[i][j].real, value.real);
                        try testing.expectEqual(original_values[i][j].imag, value.imag);
                    } else {
                        try testing.expectEqual(original_values[i][j], value);
                    }
                }
            }
        }
    }
}

test "transpose_2d_inplace - with known values" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    // Fill with known pattern: [[1, 2, 3], [4, 5, 6]]
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try memory.readFromBuffer(f32, pipeline, tensor, &input_data);

    // Transpose in-place
    try transpose_2d_inplace(f32, pipeline, tensor);

    // Shape should now be [3, 2]
    try testing.expectEqual(3, tensor.dimensions.shape[0]);
    try testing.expectEqual(2, tensor.dimensions.shape[1]);

    // Expected: [[1, 4], [2, 5], [3, 6]]
    const expected = [_][2]f32{ .{ 1, 4 }, .{ 2, 5 }, .{ 3, 6 } };
    for (0..3) |i| {
        for (0..2) |j| {
            const coords = [_]u64{ i, j };
            var value: f32 = undefined;
            try memory.getValue(f32, pipeline, tensor, &coords, &value);
            pipeline.waitAndCleanup();
            try testing.expectEqual(expected[i][j], value);
        }
    }
}

test "transpose_2d_inplace - double transpose returns original" {
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

            try random.uniform(T, pipeline, tensor, 333, null, null);

            // Save original values
            var original_values: [3][4]T = undefined;
            for (0..3) |i| {
                for (0..4) |j| {
                    const coords = [_]u64{ i, j };
                    try memory.getValue(T, pipeline, tensor, &coords, &original_values[i][j]);
                    pipeline.waitAndCleanup();
                }
            }

            // Double transpose
            try transpose_2d_inplace(T, pipeline, tensor);
            try transpose_2d_inplace(T, pipeline, tensor);

            // Shape should return to [3, 4]
            try testing.expectEqual(3, tensor.dimensions.shape[0]);
            try testing.expectEqual(4, tensor.dimensions.shape[1]);

            // Verify original values are restored
            for (0..3) |i| {
                for (0..4) |j| {
                    const coords = [_]u64{ i, j };
                    var value: T = undefined;
                    try memory.getValue(T, pipeline, tensor, &coords, &value);
                    pipeline.waitAndCleanup();

                    if (comptime core.types.isComplex(T)) {
                        try testing.expectEqual(original_values[i][j].real, value.real);
                        try testing.expectEqual(original_values[i][j].imag, value.imag);
                    } else {
                        try testing.expectEqual(original_values[i][j], value);
                    }
                }
            }
        }
    }
}

test "transpose_2d_inplace - non-2D tensor returns error" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const config = tensor_module.CreateConfig{};

    // 1D tensor
    const shape_1d = [_]u64{6};
    const tensor_1d = try Tensor(f32).alloc(context, pipeline, &shape_1d, config);
    defer tensor_1d.release(pipeline);

    try testing.expectError(tensor_module.Errors.InvalidValue, transpose_2d_inplace(f32, pipeline, tensor_1d));

    // 3D tensor
    const shape_3d = [_]u64{ 2, 3, 4 };
    const tensor_3d = try Tensor(f32).alloc(context, pipeline, &shape_3d, config);
    defer tensor_3d.release(pipeline);

    try testing.expectError(tensor_module.Errors.InvalidValue, transpose_2d_inplace(f32, pipeline, tensor_3d));
}

test "transpose - invalid dimension out of bounds" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    const result = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer result.release(pipeline);

    // Try to transpose with dimension 3 (out of bounds for 2D tensor)
    const err = transpose(f32, pipeline, result, tensor, 0, 3);
    try testing.expectError(tensor_module.Errors.InvalidValue, err);
}

test "transpose - incompatible tensor shapes" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const wrong_shape = [_]u64{ 2, 2 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer tensor.release(pipeline);

    const result = try Tensor(f32).alloc(context, pipeline, &wrong_shape, config);
    defer result.release(pipeline);

    // Try to transpose with incompatible result shape
    const err = transpose(f32, pipeline, result, tensor, 0, 1);
    try testing.expectError(tensor_module.Errors.UnqualTensorsDimension, err);
}

test "transpose - different number of dimensions" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape_2d = [_]u64{ 2, 3 };
    const shape_3d = [_]u64{ 2, 3, 4 };
    const config = tensor_module.CreateConfig{};

    const tensor = try Tensor(f32).alloc(context, pipeline, &shape_2d, config);
    defer tensor.release(pipeline);

    const result = try Tensor(f32).alloc(context, pipeline, &shape_3d, config);
    defer result.release(pipeline);

    // Try to transpose tensors with different number of dimensions
    const err = transpose(f32, pipeline, result, tensor, 0, 1);
    try testing.expectError(tensor_module.Errors.UnqualTensorsDimension, err);
}

test "transpose - double transpose returns original" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const transposed_shape = [_]u64{ 4, 3 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T)) {
            const tensor = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer tensor.release(pipeline);

            const transposed = try Tensor(T).alloc(context, pipeline, &transposed_shape, config);
            defer transposed.release(pipeline);

            const double_transposed = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer double_transposed.release(pipeline);

            // Fill with random values
            try random.uniform(T, pipeline, tensor, 333, null, null);

            // First transpose
            try transpose(T, pipeline, transposed, tensor, 0, 1);

            // Second transpose (should return to original)
            try transpose(T, pipeline, double_transposed, transposed, 0, 1);

            // Verify original and double transposed are identical
            const buffer_size = shape[0] * shape[1];
            const original_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(original_buffer);

            const double_transposed_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(double_transposed_buffer);

            try memory.writeToBuffer(T, pipeline, tensor, original_buffer);
            try memory.writeToBuffer(T, pipeline, double_transposed, double_transposed_buffer);
            pipeline.waitAndCleanup();

            for (original_buffer, double_transposed_buffer) |orig, double_trans| {
                if (comptime core.types.isComplex(T)) {
                    try testing.expectEqual(orig.real, double_trans.real);
                    try testing.expectEqual(orig.imag, double_trans.imag);
                } else {
                    try testing.expectEqual(orig, double_trans);
                }
            }
        }
    }
}

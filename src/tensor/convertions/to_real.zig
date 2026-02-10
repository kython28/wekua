const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("../main.zig");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const helpers = @import("../helpers.zig");

const to_real_cl_kernel: []const u8 = @embedFile("kernels/to_real.cl");

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    space: core.types.Space,
) TensorErrors!cl.kernel.Kernel {
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .ToReal, core.types.SUPPORTED_TYPES.len * 2);
    const index: usize = 2 * @as(usize, core.types.getTypeId(T)) + @intFromEnum(space);
    if (kernels_set.kernels.?[index]) |v| return v;

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;
    const allocator = command_queue.context.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(allocator, "-DOFFSET={d}", .{@intFromEnum(space)});
    defer allocator.free(extra_args);

    try KernelsSet.compileKernel(
        core.types.getType(T),
        command_queue,
        .{
            .vectors_enabled = false,
            .kernel_name = "to_real",
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        to_real_cl_kernel,
    );

    kernels_set.kernels.?[index] = kernel;
    kernels_set.programs.?[index] = program;

    return kernel;
}

pub fn toReal(
    comptime T: type,
    pipeline: *Pipeline,
    src: *Tensor(T),
    dst: *Tensor(core.types.getType(T)),
    space: core.types.Space,
) TensorErrors!void {
    if (!std.mem.eql(u64, src.dimensions.shape, dst.dimensions.shape)) {
        return TensorErrors.UnqualTensorsShape;
    }

    const command_queue = pipeline.command_queue;
    const kernel = try getKernel(T, command_queue, space);

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&src.buffer));
    try setArg(kernel, 1, cl_mem_size, @ptrCast(&dst.buffer));
    try setArg(kernel, 2, @sizeOf(u64), @ptrCast(&src.memory_layout.row_pitch));
    try setArg(kernel, 3, @sizeOf(u64), @ptrCast(&src.memory_layout.slice_pitch));
    try setArg(kernel, 4, @sizeOf(u64), @ptrCast(&dst.memory_layout.row_pitch));
    try setArg(kernel, 5, @sizeOf(u64), @ptrCast(&dst.memory_layout.slice_pitch));

    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        &src.work_configuration.global_work_items_without_vectors,
        &src.work_configuration.local_work_items_without_vectors[command_queue.wekua_id],
        prev_events,
        &new_event,
    );
    errdefer helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

// -----------------------------------------------------------------------------
// Unit Tests
const testing = std.testing;

const memory = tensor_module.memory;
const fill = tensor_module.fill;

test "toReal - extract real space for all complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if ((comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const RealT = core.types.getType(T);
            const dst = try Tensor(RealT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill source complex tensor with known values
            const buffer_size = shape[0] * shape[1];
            const input_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_buffer);

            const real_value: RealT = switch (@typeInfo(RealT)) {
                .int => 42,
                .float => 42.5,
                else => unreachable,
            };
            const imag_value: RealT = switch (@typeInfo(RealT)) {
                .int => 99,
                .float => 99.5,
                else => unreachable,
            };

            for (input_buffer) |*val| {
                val.* = .{ .real = real_value, .imag = imag_value };
            }

            try memory.readFromBuffer(T, pipeline, src, input_buffer);

            // Extract real space
            try toReal(T, pipeline, src, dst, .real);

            // Verify
            const output_buffer = try allocator.alloc(RealT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(RealT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            // Check all values
            for (output_buffer) |val| {
                try testing.expectEqual(real_value, val);
            }
        }
    }
}

test "toReal - extract imag space for all complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if ((comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const RealT = core.types.getType(T);
            const dst = try Tensor(RealT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill source complex tensor with known values
            const buffer_size = shape[0] * shape[1];
            const input_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_buffer);

            const real_value: RealT = switch (@typeInfo(RealT)) {
                .int => 10,
                .float => 10.5,
                else => unreachable,
            };
            const imag_value: RealT = switch (@typeInfo(RealT)) {
                .int => 77,
                .float => 77.5,
                else => unreachable,
            };

            for (input_buffer) |*val| {
                val.* = .{ .real = real_value, .imag = imag_value };
            }

            try memory.readFromBuffer(T, pipeline, src, input_buffer);

            // Extract imag space
            try toReal(T, pipeline, src, dst, .imag);

            // Verify
            const output_buffer = try allocator.alloc(RealT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(RealT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            // Check all values
            for (output_buffer) |val| {
                try testing.expectEqual(imag_value, val);
            }
        }
    }
}

test "toReal - 1D tensor extract real space for all complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if ((comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const RealT = core.types.getType(T);
            const dst = try Tensor(RealT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill source
            const buffer_size = shape[0];
            const input_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_buffer);

            const real_value: RealT = switch (@typeInfo(RealT)) {
                .int => 7,
                .float => 7.5,
                else => unreachable,
            };
            const imag_value: RealT = switch (@typeInfo(RealT)) {
                .int => 13,
                .float => 13.5,
                else => unreachable,
            };

            for (input_buffer) |*val| {
                val.* = .{ .real = real_value, .imag = imag_value };
            }

            try memory.readFromBuffer(T, pipeline, src, input_buffer);

            // Extract real space
            try toReal(T, pipeline, src, dst, .real);

            // Verify
            const output_buffer = try allocator.alloc(RealT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(RealT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            for (output_buffer) |val| {
                try testing.expectEqual(real_value, val);
            }
        }
    }
}

test "toReal - 3D tensor extract imag space for all complex types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if ((comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const RealT = core.types.getType(T);
            const dst = try Tensor(RealT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill source
            const buffer_size = shape[0] * shape[1] * shape[2];
            const input_buffer = try allocator.alloc(T, buffer_size);
            defer allocator.free(input_buffer);

            const real_value: RealT = switch (@typeInfo(RealT)) {
                .int => 5,
                .float => 5.25,
                else => unreachable,
            };
            const imag_value: RealT = switch (@typeInfo(RealT)) {
                .int => 25,
                .float => 25.75,
                else => unreachable,
            };

            for (input_buffer) |*val| {
                val.* = .{ .real = real_value, .imag = imag_value };
            }

            try memory.readFromBuffer(T, pipeline, src, input_buffer);

            // Extract imag space
            try toReal(T, pipeline, src, dst, .imag);

            // Verify
            const output_buffer = try allocator.alloc(RealT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(RealT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            for (output_buffer) |val| {
                try testing.expectEqual(imag_value, val);
            }
        }
    }
}

test "toReal - with specific values extract real space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // Fill with specific complex values
    const input_data = [_]core.types.ComplexF32{
        .{ .real = 1.0, .imag = 10.0 },
        .{ .real = 2.0, .imag = 20.0 },
        .{ .real = 3.0, .imag = 30.0 },
        .{ .real = 4.0, .imag = 40.0 },
        .{ .real = 5.0, .imag = 50.0 },
    };
    try memory.readFromBuffer(core.types.ComplexF32, pipeline, src, &input_data);

    // Extract real space
    try toReal(core.types.ComplexF32, pipeline, src, dst, .real);

    // Verify
    const output_buffer = try allocator.alloc(f32, 5);
    defer allocator.free(output_buffer);

    try memory.writeToBuffer(f32, pipeline, dst, output_buffer);
    pipeline.waitAndCleanup();

    const expected = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    for (output_buffer, expected) |actual, exp| {
        try testing.expectEqual(exp, actual);
    }
}

test "toReal - with specific values extract imag space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // Fill with specific complex values
    const input_data = [_]core.types.ComplexF32{
        .{ .real = 1.0, .imag = 100.0 },
        .{ .real = 2.0, .imag = 200.0 },
        .{ .real = 3.0, .imag = 300.0 },
        .{ .real = 4.0, .imag = 400.0 },
        .{ .real = 5.0, .imag = 500.0 },
    };
    try memory.readFromBuffer(core.types.ComplexF32, pipeline, src, &input_data);

    // Extract imag space
    try toReal(core.types.ComplexF32, pipeline, src, dst, .imag);

    // Verify
    const output_buffer = try allocator.alloc(f32, 5);
    defer allocator.free(output_buffer);

    try memory.writeToBuffer(f32, pipeline, dst, output_buffer);
    pipeline.waitAndCleanup();

    const expected = [_]f32{ 100.0, 200.0, 300.0, 400.0, 500.0 };
    for (output_buffer, expected) |actual, exp| {
        try testing.expectEqual(exp, actual);
    }
}

test "toReal - verify using getValue for real space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // Fill with pattern
    const input_data = [_]core.types.ComplexF32{
        .{ .real = 1.0, .imag = 10.0 },
        .{ .real = 2.0, .imag = 20.0 },
        .{ .real = 3.0, .imag = 30.0 },
        .{ .real = 4.0, .imag = 40.0 },
        .{ .real = 5.0, .imag = 50.0 },
        .{ .real = 6.0, .imag = 60.0 },
    };
    try memory.readFromBuffer(core.types.ComplexF32, pipeline, src, &input_data);

    // Extract real space
    try toReal(core.types.ComplexF32, pipeline, src, dst, .real);

    // Verify using getValue
    var idx: usize = 0;
    for (0..shape[0]) |i| {
        for (0..shape[1]) |j| {
            const coords = [_]u64{ i, j };
            var value: f32 = undefined;

            try memory.getValue(f32, pipeline, dst, &coords, &value);
            pipeline.waitAndCleanup();

            try testing.expectEqual(input_data[idx].real, value);
            idx += 1;
        }
    }
}

test "toReal - verify using getValue for imag space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // Fill with pattern
    const input_data = [_]core.types.ComplexF32{
        .{ .real = 1.0, .imag = 100.0 },
        .{ .real = 2.0, .imag = 200.0 },
        .{ .real = 3.0, .imag = 300.0 },
        .{ .real = 4.0, .imag = 400.0 },
        .{ .real = 5.0, .imag = 500.0 },
        .{ .real = 6.0, .imag = 600.0 },
    };
    try memory.readFromBuffer(core.types.ComplexF32, pipeline, src, &input_data);

    // Extract imag space
    try toReal(core.types.ComplexF32, pipeline, src, dst, .imag);

    // Verify using getValue
    var idx: usize = 0;
    for (0..shape[0]) |i| {
        for (0..shape[1]) |j| {
            const coords = [_]u64{ i, j };
            var value: f32 = undefined;

            try memory.getValue(f32, pipeline, dst, &coords, &value);
            pipeline.waitAndCleanup();

            try testing.expectEqual(input_data[idx].imag, value);
            idx += 1;
        }
    }
}

test "toReal - round trip with toComplex" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const to_complex = @import("to_complex.zig").toComplex;

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const original = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer original.release(pipeline);

            const ComplexT = core.types.Complex(T);
            const complex = try Tensor(ComplexT).alloc(context, pipeline, &shape, config);
            defer complex.release(pipeline);

            const result = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer result.release(pipeline);

            // Fill original with values
            const original_data = try allocator.alloc(T, shape[0]);
            defer allocator.free(original_data);

            for (original_data, 0..) |*val, i| {
                val.* = switch (@typeInfo(T)) {
                    .int => @intCast(i * 3),
                    .float => @as(T, @floatFromInt(i)) * 3.5,
                    else => unreachable,
                };
            }

            try memory.readFromBuffer(T, pipeline, original, original_data);

            // Convert to complex (real space)
            try to_complex(T, pipeline, original, complex, .real);

            // Convert back to real (real space)
            try toReal(ComplexT, pipeline, complex, result, .real);

            // Verify round trip
            const result_data = try allocator.alloc(T, shape[0]);
            defer allocator.free(result_data);

            try memory.writeToBuffer(T, pipeline, result, result_data);
            pipeline.waitAndCleanup();

            for (original_data, result_data) |expected, actual| {
                try testing.expectEqual(expected, actual);
            }
        }
    }
}

test "toReal - incompatible shapes error" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape1 = [_]u64{ 2, 3 };
    const shape2 = [_]u64{ 3, 2 };
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape1, config);
    defer src.release(pipeline);

    const dst = try Tensor(f32).alloc(context, pipeline, &shape2, config);
    defer dst.release(pipeline);

    // Try to convert with incompatible shapes
    const err = toReal(core.types.ComplexF32, pipeline, src, dst, .real);
    try testing.expectError(tensor_module.Errors.UnqualTensorsShape, err);
}

test "toReal - different number of dimensions error" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape_2d = [_]u64{ 2, 3 };
    const shape_3d = [_]u64{ 2, 3, 1 };
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape_2d, config);
    defer src.release(pipeline);

    const dst = try Tensor(f32).alloc(context, pipeline, &shape_3d, config);
    defer dst.release(pipeline);

    // Try to convert with different dimensions
    const err = toReal(core.types.ComplexF32, pipeline, src, dst, .real);
    try testing.expectError(tensor_module.Errors.UnqualTensorsShape, err);
}

test "toReal - overwrite previous data extract real" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // First extraction
    const first_data = [_]core.types.ComplexF32{
        .{ .real = 1.0, .imag = 10.0 },
        .{ .real = 2.0, .imag = 20.0 },
        .{ .real = 3.0, .imag = 30.0 },
        .{ .real = 4.0, .imag = 40.0 },
        .{ .real = 5.0, .imag = 50.0 },
    };
    try memory.readFromBuffer(core.types.ComplexF32, pipeline, src, &first_data);
    try toReal(core.types.ComplexF32, pipeline, src, dst, .real);

    // Second extraction (should overwrite)
    const second_data = [_]core.types.ComplexF32{
        .{ .real = 100.0, .imag = 1000.0 },
        .{ .real = 200.0, .imag = 2000.0 },
        .{ .real = 300.0, .imag = 3000.0 },
        .{ .real = 400.0, .imag = 4000.0 },
        .{ .real = 500.0, .imag = 5000.0 },
    };
    try memory.readFromBuffer(core.types.ComplexF32, pipeline, src, &second_data);
    try toReal(core.types.ComplexF32, pipeline, src, dst, .real);

    // Verify
    const output_buffer = try allocator.alloc(f32, 5);
    defer allocator.free(output_buffer);

    try memory.writeToBuffer(f32, pipeline, dst, output_buffer);
    pipeline.waitAndCleanup();

    const expected = [_]f32{ 100.0, 200.0, 300.0, 400.0, 500.0 };
    for (output_buffer, expected) |actual, exp| {
        try testing.expectEqual(exp, actual);
    }
}

test "toReal - overwrite previous data extract imag" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(f32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // First extraction
    const first_data = [_]core.types.ComplexF32{
        .{ .real = 1.0, .imag = 10.0 },
        .{ .real = 2.0, .imag = 20.0 },
        .{ .real = 3.0, .imag = 30.0 },
        .{ .real = 4.0, .imag = 40.0 },
        .{ .real = 5.0, .imag = 50.0 },
    };
    try memory.readFromBuffer(core.types.ComplexF32, pipeline, src, &first_data);
    try toReal(core.types.ComplexF32, pipeline, src, dst, .imag);

    // Second extraction (should overwrite)
    const second_data = [_]core.types.ComplexF32{
        .{ .real = 100.0, .imag = 1000.0 },
        .{ .real = 200.0, .imag = 2000.0 },
        .{ .real = 300.0, .imag = 3000.0 },
        .{ .real = 400.0, .imag = 4000.0 },
        .{ .real = 500.0, .imag = 5000.0 },
    };
    try memory.readFromBuffer(core.types.ComplexF32, pipeline, src, &second_data);
    try toReal(core.types.ComplexF32, pipeline, src, dst, .imag);

    // Verify
    const output_buffer = try allocator.alloc(f32, 5);
    defer allocator.free(output_buffer);

    try memory.writeToBuffer(f32, pipeline, dst, output_buffer);
    pipeline.waitAndCleanup();

    const expected = [_]f32{ 1000.0, 2000.0, 3000.0, 4000.0, 5000.0 };
    for (output_buffer, expected) |actual, exp| {
        try testing.expectEqual(exp, actual);
    }
}

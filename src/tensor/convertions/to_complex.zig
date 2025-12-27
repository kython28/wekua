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

const to_complex_cl_kernel: []const u8 = @embedFile("kernels/to_complex.cl");

fn getKernel(
    comptime T: type,
    command_queue: *const CommandQueue,
    space: core.types.Space,
) TensorErrors!cl.kernel.Kernel {
    const kernels_set = try KernelsSet.getKernelSet(command_queue, .ToComplex, core.types.SUPPORTED_TYPES.len * 2);
    const index: usize = 2 * @as(usize, core.types.getTypeId(T)) + @intFromEnum(space);
    if (kernels_set.kernels.?[index]) |v| return v;

    var kernel: cl.kernel.Kernel = undefined;
    var program: cl.program.Program = undefined;

    const allocator = command_queue.context.allocator;
    const extra_args: []u8 = try std.fmt.allocPrint(allocator, "-DOFFSET={d}", .{@intFromEnum(space)});
    defer allocator.free(extra_args);

    try KernelsSet.compileKernel(
        T,
        command_queue,
        .{
            .vectors_enabled = false,
            .kernel_name = "to_complex",
            .extra_args = extra_args,
        },
        &kernel,
        &program,
        to_complex_cl_kernel,
    );

    kernels_set.kernels.?[index] = kernel;
    kernels_set.programs.?[index] = program;

    return kernel;
}

pub fn toComplex(
    comptime T: type,
    pipeline: *Pipeline,
    src: *Tensor(T),
    dst: *Tensor(core.types.Complex(T)),
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

test "toComplex - real space for all real types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const ComplexT = core.types.Complex(T);
            const dst = try Tensor(ComplexT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill source with known value
            const value: T = switch (@typeInfo(T)) {
                .int => 42,
                .float => 42.5,
                else => unreachable,
            };
            try fill.constant(T, pipeline, src, value);

            // Convert to complex (real space)
            try toComplex(T, pipeline, src, dst, .real);

            // Verify
            const buffer_size = shape[0] * shape[1];
            const output_buffer = try allocator.alloc(ComplexT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(ComplexT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            // Check all values
            for (output_buffer) |val| {
                try testing.expectEqual(value, val.real);
                try testing.expectEqual(@as(T, 0), val.imag);
            }
        }
    }
}

test "toComplex - imag space for all real types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const ComplexT = core.types.Complex(T);
            const dst = try Tensor(ComplexT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill source with known value
            const value: T = switch (@typeInfo(T)) {
                .int => 99,
                .float => 99.5,
                else => unreachable,
            };
            try fill.constant(T, pipeline, src, value);

            // Convert to complex (imag space)
            try toComplex(T, pipeline, src, dst, .imag);

            // Verify
            const buffer_size = shape[0] * shape[1];
            const output_buffer = try allocator.alloc(ComplexT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(ComplexT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            // Check all values
            for (output_buffer) |val| {
                try testing.expectEqual(@as(T, 0), val.real);
                try testing.expectEqual(value, val.imag);
            }
        }
    }
}

test "toComplex - 1D tensor to real space for all real types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{10};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const ComplexT = core.types.Complex(T);
            const dst = try Tensor(ComplexT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill source
            const value: T = switch (@typeInfo(T)) {
                .int => 7,
                .float => 7.5,
                else => unreachable,
            };
            try fill.constant(T, pipeline, src, value);

            // Convert to complex (real space)
            try toComplex(T, pipeline, src, dst, .real);

            // Verify
            const buffer_size = shape[0];
            const output_buffer = try allocator.alloc(ComplexT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(ComplexT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            for (output_buffer) |val| {
                try testing.expectEqual(value, val.real);
                try testing.expectEqual(@as(T, 0), val.imag);
            }
        }
    }
}

test "toComplex - 3D tensor to imag space for all real types" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3, 4 };
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src.release(pipeline);

            const ComplexT = core.types.Complex(T);
            const dst = try Tensor(ComplexT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill source
            const value: T = switch (@typeInfo(T)) {
                .int => 13,
                .float => 13.25,
                else => unreachable,
            };
            try fill.constant(T, pipeline, src, value);

            // Convert to complex (imag space)
            try toComplex(T, pipeline, src, dst, .imag);

            // Verify
            const buffer_size = shape[0] * shape[1] * shape[2];
            const output_buffer = try allocator.alloc(ComplexT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(ComplexT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            for (output_buffer) |val| {
                try testing.expectEqual(@as(T, 0), val.real);
                try testing.expectEqual(value, val.imag);
            }
        }
    }
}

test "toComplex - with specific values to real space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(f32).empty(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // Fill with specific values
    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try memory.readFromBuffer(f32, pipeline, src, &input_data);

    // Convert to complex (real space)
    try toComplex(f32, pipeline, src, dst, .real);

    // Verify
    const output_buffer = try allocator.alloc(core.types.ComplexF32, 5);
    defer allocator.free(output_buffer);

    try memory.writeToBuffer(core.types.ComplexF32, pipeline, dst, output_buffer);
    pipeline.waitAndCleanup();

    for (input_data, output_buffer) |expected, actual| {
        try testing.expectEqual(expected, actual.real);
        try testing.expectEqual(@as(f32, 0), actual.imag);
    }
}

test "toComplex - with specific values to imag space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(f32).empty(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // Fill with specific values
    const input_data = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    try memory.readFromBuffer(f32, pipeline, src, &input_data);

    // Convert to complex (imag space)
    try toComplex(f32, pipeline, src, dst, .imag);

    // Verify
    const output_buffer = try allocator.alloc(core.types.ComplexF32, 5);
    defer allocator.free(output_buffer);

    try memory.writeToBuffer(core.types.ComplexF32, pipeline, dst, output_buffer);
    pipeline.waitAndCleanup();

    for (input_data, output_buffer) |expected, actual| {
        try testing.expectEqual(@as(f32, 0), actual.real);
        try testing.expectEqual(expected, actual.imag);
    }
}

test "toComplex - combining real and imag conversions" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{4};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (!(comptime core.types.isComplex(T)) and command_queue.isTypeSupported(T)) {
            const src_real = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src_real.release(pipeline);

            const src_imag = try Tensor(T).alloc(context, pipeline, &shape, config);
            defer src_imag.release(pipeline);

            const ComplexT = core.types.Complex(T);
            const dst = try Tensor(ComplexT).alloc(context, pipeline, &shape, config);
            defer dst.release(pipeline);

            // Fill sources
            const real_value: T = switch (@typeInfo(T)) {
                .int => 5,
                .float => 5.5,
                else => unreachable,
            };
            const imag_value: T = switch (@typeInfo(T)) {
                .int => 10,
                .float => 10.5,
                else => unreachable,
            };

            try fill.constant(T, pipeline, src_real, real_value);
            try fill.constant(T, pipeline, src_imag, imag_value);

            // Convert real part
            try toComplex(T, pipeline, src_real, dst, .real);

            // Convert imag part
            try toComplex(T, pipeline, src_imag, dst, .imag);

            // Verify
            const buffer_size = shape[0];
            const output_buffer = try allocator.alloc(ComplexT, buffer_size);
            defer allocator.free(output_buffer);

            try memory.writeToBuffer(ComplexT, pipeline, dst, output_buffer);
            pipeline.waitAndCleanup();

            for (output_buffer) |val| {
                try testing.expectEqual(real_value, val.real);
                try testing.expectEqual(imag_value, val.imag);
            }
        }
    }
}

test "toComplex - verify using getValue for real space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(f32).empty(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // Fill with pattern
    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try memory.readFromBuffer(f32, pipeline, src, &input_data);

    // Convert to complex (real space)
    try toComplex(f32, pipeline, src, dst, .real);

    // Verify using getValue
    var idx: usize = 0;
    for (0..shape[0]) |i| {
        for (0..shape[1]) |j| {
            const coords = [_]u64{ i, j };
            var value: core.types.ComplexF32 = undefined;

            try memory.getValue(core.types.ComplexF32, pipeline, dst, &coords, &value);
            pipeline.waitAndCleanup();

            try testing.expectEqual(input_data[idx], value.real);
            try testing.expectEqual(@as(f32, 0), value.imag);
            idx += 1;
        }
    }
}

test "toComplex - verify using getValue for imag space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{ 2, 3 };
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(f32).empty(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // Fill with pattern
    const input_data = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 };
    try memory.readFromBuffer(f32, pipeline, src, &input_data);

    // Convert to complex (imag space)
    try toComplex(f32, pipeline, src, dst, .imag);

    // Verify using getValue
    var idx: usize = 0;
    for (0..shape[0]) |i| {
        for (0..shape[1]) |j| {
            const coords = [_]u64{ i, j };
            var value: core.types.ComplexF32 = undefined;

            try memory.getValue(core.types.ComplexF32, pipeline, dst, &coords, &value);
            pipeline.waitAndCleanup();

            try testing.expectEqual(@as(f32, 0), value.real);
            try testing.expectEqual(input_data[idx], value.imag);
            idx += 1;
        }
    }
}

test "toComplex - incompatible shapes error" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape1 = [_]u64{ 2, 3 };
    const shape2 = [_]u64{ 3, 2 };
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(f32).alloc(context, pipeline, &shape1, config);
    defer src.release(pipeline);

    const dst = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape2, config);
    defer dst.release(pipeline);

    // Try to convert with incompatible shapes
    const err = toComplex(f32, pipeline, src, dst, .real);
    try testing.expectError(tensor_module.Errors.UnqualTensorsShape, err);
}

test "toComplex - different number of dimensions error" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape_2d = [_]u64{ 2, 3 };
    const shape_3d = [_]u64{ 2, 3, 1 };
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(f32).alloc(context, pipeline, &shape_2d, config);
    defer src.release(pipeline);

    const dst = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape_3d, config);
    defer dst.release(pipeline);

    // Try to convert with different dimensions
    const err = toComplex(f32, pipeline, src, dst, .real);
    try testing.expectError(tensor_module.Errors.UnqualTensorsShape, err);
}

test "toComplex - overwrite previous data in real space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(f32).empty(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // First conversion
    const first_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try memory.readFromBuffer(f32, pipeline, src, &first_data);
    try toComplex(f32, pipeline, src, dst, .real);

    // Second conversion (should overwrite)
    const second_data = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    try memory.readFromBuffer(f32, pipeline, src, &second_data);
    try toComplex(f32, pipeline, src, dst, .real);

    // Verify
    const output_buffer = try allocator.alloc(core.types.ComplexF32, 5);
    defer allocator.free(output_buffer);

    try memory.writeToBuffer(core.types.ComplexF32, pipeline, dst, output_buffer);
    pipeline.waitAndCleanup();

    for (second_data, output_buffer) |expected, actual| {
        try testing.expectEqual(expected, actual.real);
        try testing.expectEqual(@as(f32, 0), actual.imag);
    }
}

test "toComplex - overwrite previous data in imag space" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{5};
    const config = tensor_module.CreateConfig{};

    const src = try Tensor(f32).empty(context, pipeline, &shape, config);
    defer src.release(pipeline);

    const dst = try Tensor(core.types.ComplexF32).alloc(context, pipeline, &shape, config);
    defer dst.release(pipeline);

    // First conversion
    const first_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try memory.readFromBuffer(f32, pipeline, src, &first_data);
    try toComplex(f32, pipeline, src, dst, .imag);

    // Second conversion (should overwrite)
    const second_data = [_]f32{ 100.0, 200.0, 300.0, 400.0, 500.0 };
    try memory.readFromBuffer(f32, pipeline, src, &second_data);
    try toComplex(f32, pipeline, src, dst, .imag);

    // Verify
    const output_buffer = try allocator.alloc(core.types.ComplexF32, 5);
    defer allocator.free(output_buffer);

    try memory.writeToBuffer(core.types.ComplexF32, pipeline, dst, output_buffer);
    pipeline.waitAndCleanup();

    for (second_data, output_buffer) |expected, actual| {
        try testing.expectEqual(@as(f32, 0), actual.real);
        try testing.expectEqual(expected, actual.imag);
    }
}

const std = @import("std");
const cl = @import("opencl");

const core = @import("core");
const Pipeline = core.Pipeline;
const CommandQueue = core.CommandQueue;
const KernelsSet = core.KernelsSet;

const tensor_module = @import("tensor");
const Tensor = tensor_module.Tensor;
const TensorErrors = tensor_module.Errors;

const trigonometric_cl_kernel: []const u8 = @embedFile("kernels/trig.cl");

fn genericTrigFunction(
    comptime T: type,
    kernel_name: []const u8,
    kernel_id: KernelsSet.KernelsID,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    const command_queue = pipeline.command_queue;

    const kernel = try KernelsSet.getClKernel(
        T,
        command_queue,
        tensor.flags.vectors_enabled,
        kernel_id,
        kernel_name,
        trigonometric_cl_kernel,
        null,
    );

    const prev_events = pipeline.prevEvents();

    const setArg = cl.kernel.setArg;
    const cl_mem_size = @sizeOf(cl.buffer.Mem);

    try setArg(kernel, 0, cl_mem_size, @ptrCast(&tensor.buffer));

    const wekua_id = command_queue.wekua_id;
    var global_work_items: [1]u64 = undefined;
    var local_work_items: []const u64 = undefined;

    if (tensor.flags.vectors_enabled) {
        global_work_items = .{tensor.memory_layout.number_of_vectors};
        local_work_items = tensor.work_configuration.local_work_items_for_vectors_1d[wekua_id .. wekua_id + 1];
    } else {
        global_work_items = .{tensor.dimensions.number_of_elements};
        local_work_items = tensor.work_configuration.local_work_items_1d[wekua_id .. wekua_id + 1];
    }

    var new_event: cl.event.Event = undefined;
    try cl.kernel.enqueueNdRange(
        command_queue.cl_command_queue,
        kernel,
        null,
        &global_work_items,
        local_work_items,
        prev_events,
        &new_event,
    );
    errdefer tensor_module.helpers.releaseEvent(new_event);

    try pipeline.append(&.{new_event});
}

pub inline fn sin(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    try genericTrigFunction(T, "sin_kernel", .Sin, pipeline, tensor);
}

pub inline fn cos(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    try genericTrigFunction(T, "cos_kernel", .Cos, pipeline, tensor);
}

pub inline fn tan(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    try genericTrigFunction(T, "tan_kernel", .Tan, pipeline, tensor);
}

pub inline fn sinh(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    try genericTrigFunction(T, "sinh_kernel", .Sinh, pipeline, tensor);
}

pub inline fn cosh(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    try genericTrigFunction(T, "cosh_kernel", .Cosh, pipeline, tensor);
}

pub inline fn tanh(
    comptime T: type,
    pipeline: *Pipeline,
    tensor: *Tensor(T),
) TensorErrors!void {
    try genericTrigFunction(T, "tanh_kernel", .Tanh, pipeline, tensor);
}

// -----------------------------------------------------------------------------
// Unit Tests
const testing = std.testing;

const memory = tensor_module.memory;

fn isFloatType(comptime T: type) bool {
    if (comptime core.types.isComplex(T)) {
        return @typeInfo(core.types.getType(T)) == .float;
    }
    return @typeInfo(T) == .float;
}

test "sin - basic operation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{4};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T) and comptime isFloatType(T)) {
            const x = try Tensor(T).empty(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(buf);

            // sin(0) = 0, sin(pi/6) ≈ 0.5, sin(pi/2) ≈ 1, sin(pi) ≈ 0
            if (comptime core.types.isComplex(T)) {
                const SubType = core.types.getType(T);
                buf[0] = .{ .real = 0, .imag = 0 };
                buf[1] = .{ .real = @as(SubType, std.math.pi) / 6.0, .imag = 0 };
                buf[2] = .{ .real = @as(SubType, std.math.pi) / 2.0, .imag = 0 };
                buf[3] = .{ .real = std.math.pi, .imag = 0 };
            } else {
                buf[0] = 0;
                buf[1] = @as(T, std.math.pi) / 6.0;
                buf[2] = @as(T, std.math.pi) / 2.0;
                buf[3] = std.math.pi;
            }

            try memory.readFromBuffer(T, pipeline, x, buf);
            try sin(T, pipeline, x);

            const result = try allocator.alloc(T, shape[0]);
            defer allocator.free(result);

            try memory.writeToBuffer(T, pipeline, x, result);
            pipeline.waitAndCleanup();

            const eps: f64 = 1e-5;
            if (comptime core.types.isComplex(T)) {
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[0].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, 0.5), @as(f64, @floatCast(result[1].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, 1.0), @as(f64, @floatCast(result[2].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[3].real)), eps);
            } else {
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[0])), eps);
                try testing.expectApproxEqAbs(@as(f64, 0.5), @as(f64, @floatCast(result[1])), eps);
                try testing.expectApproxEqAbs(@as(f64, 1.0), @as(f64, @floatCast(result[2])), eps);
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[3])), eps);
            }
        }
    }
}

test "cos - basic operation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{3};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T) and comptime isFloatType(T)) {
            const x = try Tensor(T).empty(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(buf);

            // cos(0) = 1, cos(pi/2) ≈ 0, cos(pi) = -1
            if (comptime core.types.isComplex(T)) {
                buf[0] = .{ .real = 0, .imag = 0 };
                buf[1] = .{ .real = @as(core.types.getType(T), std.math.pi) / 2.0, .imag = 0 };
                buf[2] = .{ .real = std.math.pi, .imag = 0 };
            } else {
                buf[0] = 0;
                buf[1] = @as(T, std.math.pi) / 2.0;
                buf[2] = std.math.pi;
            }

            try memory.readFromBuffer(T, pipeline, x, buf);
            try cos(T, pipeline, x);

            const result = try allocator.alloc(T, shape[0]);
            defer allocator.free(result);

            try memory.writeToBuffer(T, pipeline, x, result);
            pipeline.waitAndCleanup();

            const eps: f64 = 1e-5;
            if (comptime core.types.isComplex(T)) {
                try testing.expectApproxEqAbs(@as(f64, 1), @as(f64, @floatCast(result[0].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[1].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, -1), @as(f64, @floatCast(result[2].real)), eps);
            } else {
                try testing.expectApproxEqAbs(@as(f64, 1), @as(f64, @floatCast(result[0])), eps);
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[1])), eps);
                try testing.expectApproxEqAbs(@as(f64, -1), @as(f64, @floatCast(result[2])), eps);
            }
        }
    }
}

test "tan - basic operation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{2};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T) and comptime isFloatType(T)) {
            const x = try Tensor(T).empty(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(buf);

            // tan(0) = 0, tan(pi/4) ≈ 1
            if (comptime core.types.isComplex(T)) {
                buf[0] = .{ .real = 0, .imag = 0 };
                buf[1] = .{ .real = @as(core.types.getType(T), std.math.pi) / 4.0, .imag = 0 };
            } else {
                buf[0] = 0;
                buf[1] = @as(T, std.math.pi) / 4.0;
            }

            try memory.readFromBuffer(T, pipeline, x, buf);
            try tan(T, pipeline, x);

            const result = try allocator.alloc(T, shape[0]);
            defer allocator.free(result);

            try memory.writeToBuffer(T, pipeline, x, result);
            pipeline.waitAndCleanup();

            const eps: f64 = 1e-5;
            if (comptime core.types.isComplex(T)) {
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[0].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, 1), @as(f64, @floatCast(result[1].real)), eps);
            } else {
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[0])), eps);
                try testing.expectApproxEqAbs(@as(f64, 1), @as(f64, @floatCast(result[1])), eps);
            }
        }
    }
}

test "sinh - basic operation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{2};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T) and comptime isFloatType(T)) {
            const x = try Tensor(T).empty(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(buf);

            // sinh(0) = 0, sinh(1) ≈ 1.1752
            if (comptime core.types.isComplex(T)) {
                buf[0] = .{ .real = 0, .imag = 0 };
                buf[1] = .{ .real = 1, .imag = 0 };
            } else {
                buf[0] = 0;
                buf[1] = 1;
            }

            try memory.readFromBuffer(T, pipeline, x, buf);
            try sinh(T, pipeline, x);

            const result = try allocator.alloc(T, shape[0]);
            defer allocator.free(result);

            try memory.writeToBuffer(T, pipeline, x, result);
            pipeline.waitAndCleanup();

            const eps: f64 = 1e-4;
            if (comptime core.types.isComplex(T)) {
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[0].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, 1.1752), @as(f64, @floatCast(result[1].real)), eps);
            } else {
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[0])), eps);
                try testing.expectApproxEqAbs(@as(f64, 1.1752), @as(f64, @floatCast(result[1])), eps);
            }
        }
    }
}

test "cosh - basic operation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{2};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T) and comptime isFloatType(T)) {
            const x = try Tensor(T).empty(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(buf);

            // cosh(0) = 1, cosh(1) ≈ 1.5431
            if (comptime core.types.isComplex(T)) {
                buf[0] = .{ .real = 0, .imag = 0 };
                buf[1] = .{ .real = 1, .imag = 0 };
            } else {
                buf[0] = 0;
                buf[1] = 1;
            }

            try memory.readFromBuffer(T, pipeline, x, buf);
            try cosh(T, pipeline, x);

            const result = try allocator.alloc(T, shape[0]);
            defer allocator.free(result);

            try memory.writeToBuffer(T, pipeline, x, result);
            pipeline.waitAndCleanup();

            const eps: f64 = 1e-4;
            if (comptime core.types.isComplex(T)) {
                try testing.expectApproxEqAbs(@as(f64, 1), @as(f64, @floatCast(result[0].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, 1.5431), @as(f64, @floatCast(result[1].real)), eps);
            } else {
                try testing.expectApproxEqAbs(@as(f64, 1), @as(f64, @floatCast(result[0])), eps);
                try testing.expectApproxEqAbs(@as(f64, 1.5431), @as(f64, @floatCast(result[1])), eps);
            }
        }
    }
}

test "tanh - basic operation" {
    const allocator = testing.allocator;

    const context = try core.Context.initFromDeviceType(allocator, null, cl.device.Type.all);
    defer context.deinit();

    const command_queue = &context.command_queues[0];
    const pipeline = try Pipeline.init(command_queue);
    defer pipeline.deinit();

    const shape = [_]u64{2};
    const config = tensor_module.CreateConfig{};

    inline for (core.types.SUPPORTED_TYPES) |T| {
        if (command_queue.isTypeSupported(T) and comptime isFloatType(T)) {
            const x = try Tensor(T).empty(context, pipeline, &shape, config);
            defer x.release(pipeline);

            const buf = try allocator.alloc(T, shape[0]);
            defer allocator.free(buf);

            // tanh(0) = 0, tanh(1) ≈ 0.7616
            if (comptime core.types.isComplex(T)) {
                buf[0] = .{ .real = 0, .imag = 0 };
                buf[1] = .{ .real = 1, .imag = 0 };
            } else {
                buf[0] = 0;
                buf[1] = 1;
            }

            try memory.readFromBuffer(T, pipeline, x, buf);
            try tanh(T, pipeline, x);

            const result = try allocator.alloc(T, shape[0]);
            defer allocator.free(result);

            try memory.writeToBuffer(T, pipeline, x, result);
            pipeline.waitAndCleanup();

            const eps: f64 = 1e-4;
            if (comptime core.types.isComplex(T)) {
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[0].real)), eps);
                try testing.expectApproxEqAbs(@as(f64, 0.7616), @as(f64, @floatCast(result[1].real)), eps);
            } else {
                try testing.expectApproxEqAbs(@as(f64, 0), @as(f64, @floatCast(result[0])), eps);
                try testing.expectApproxEqAbs(@as(f64, 0.7616), @as(f64, @floatCast(result[1])), eps);
            }
        }
    }
}
